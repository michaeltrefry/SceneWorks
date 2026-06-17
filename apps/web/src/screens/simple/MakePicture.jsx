import React, { useMemo, useState } from "react";
import { useAppContext } from "../../context/AppContext.js";
import { Icon } from "../../components/Icons.jsx";
import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { pickClosestResolution, parseResolution } from "../../resolutionMatch.js";
import { LookTile } from "./LookTile.jsx";
import { useLookExemplars } from "./useLookExemplars.js";
import {
  LOOKS,
  SHAPES,
  COUNT_OPTIONS,
  UPSCALE_OPTIONS,
  FALLBACK_RESOLUTIONS,
  composePrompt,
} from "./simpleDefaults.js";

// Snap a friendly shape to the closest resolution the chosen model actually
// supports, so Simple never sends an unsupported WxH.
function resolveDims(model, shape) {
  const options = model?.limits?.resolutions?.length ? model.limits.resolutions : FALLBACK_RESOLUTIONS;
  const picked = pickClosestResolution(shape.width, shape.height, options) ?? options[0] ?? "1024x1024";
  const dims = parseResolution(picked) ?? { width: 1024, height: 1024 };
  return { ...dims, resolution: `${dims.width}x${dims.height}` };
}

export function MakePicture() {
  const {
    activeProject,
    createImageJob,
    refinePrompt,
    imageModels = [],
    recentImageAssets = [],
    imageLocalJobs = [],
    setPreviewAsset,
  } = useAppContext();

  const [prompt, setPrompt] = useState("");
  const [lookId, setLookId] = useState(null);
  const [shapeId, setShapeId] = useState("square");
  const [count, setCount] = useState(4);
  const [upscaleId, setUpscaleId] = useState("off");
  const [submitting, setSubmitting] = useState(false);
  const [describing, setDescribing] = useState(false);
  const [notice, setNotice] = useState("");

  const looks = useLookExemplars();
  const model = imageModels[0] ?? null;
  const modelId = model?.id ?? "z_image_turbo";
  const look = useMemo(() => LOOKS.find((entry) => entry.id === lookId) ?? null, [lookId]);
  const shape = useMemo(() => SHAPES.find((entry) => entry.id === shapeId) ?? SHAPES[0], [shapeId]);
  const upscale = useMemo(() => UPSCALE_OPTIONS.find((entry) => entry.id === upscaleId) ?? UPSCALE_OPTIONS[0], [upscaleId]);

  const rendering = imageLocalJobs.length > 0;
  const canSubmit = Boolean(activeProject) && prompt.trim().length > 0 && !submitting;

  async function handleCreate() {
    if (!canSubmit) {
      if (!activeProject) setNotice("Open or create a workspace first.");
      return;
    }
    setSubmitting(true);
    setNotice("");
    const dims = resolveDims(model, shape);
    const payload = {
      mode: "text_to_image",
      prompt: composePrompt(prompt, look),
      negativePrompt: "",
      model: modelId,
      count,
      width: dims.width,
      height: dims.height,
      recipePresetId: look?.presetId ?? null,
      loras: [],
      advanced: { resolution: dims.resolution },
    };
    if (upscale.factor) {
      payload.upscale = { enabled: true, factor: upscale.factor };
    }
    try {
      const job = await createImageJob(payload);
      if (!job) setNotice("Couldn't start that generation — check the workspace and try again.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDescribe() {
    const base = prompt.trim();
    if (!base || describing) return;
    setDescribing(true);
    setNotice("");
    try {
      const refined = await refinePrompt({ prompt: base, modelId });
      if (refined) setPrompt(refined);
    } catch (error) {
      setNotice(error?.message || "Couldn't reach the description helper. Is its model installed?");
    } finally {
      setDescribing(false);
    }
  }

  return (
    <section className="main-surface sw-make">
      <div className="sw-make-grid">
        <div className="sw-card">
          <h3 className="sw-q">What do you want to make?</h3>
          <textarea
            className="sw-prompt"
            rows={3}
            value={prompt}
            placeholder="A cosy cabin in the snow at dusk, warm lights glowing in the windows"
            onChange={(event) => setPrompt(event.target.value)}
          />
          <div className="sw-row-actions">
            <button
              type="button"
              className="sw-chip sw-chip-spark"
              onClick={handleDescribe}
              disabled={!prompt.trim() || describing}
            >
              <Icon.Sparkle />
              <span>{describing ? "Describing…" : "Help me describe it"}</span>
            </button>
          </div>

          <div className="sw-field">
            <h3 className="sw-q sw-look-head">
              <span>Pick a look <span className="sw-opt">— optional</span></span>
              {looks.canRender ? (
                <button type="button" className="sw-refresh" onClick={() => looks.refresh()} disabled={looks.refreshing}>
                  <Icon.Sparkle /> {looks.refreshing ? "Rendering…" : looks.hasAny ? "Refresh looks" : "Preview looks"}
                </button>
              ) : null}
            </h3>
            <div className="sw-tiles">
              {LOOKS.map((entry) => (
                <button
                  type="button"
                  key={entry.id}
                  className={`sw-tile ${entry.tone} ${lookId === entry.id ? "on" : ""}`.trim()}
                  aria-pressed={lookId === entry.id}
                  onClick={() => setLookId((current) => (current === entry.id ? null : entry.id))}
                >
                  <LookTile asset={looks.assetForLook(entry.id)} pending={Boolean(looks.pending[entry.id])} />
                  <span>{entry.label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="sw-field">
            <h3 className="sw-q">Shape</h3>
            <div className="sw-chips">
              {SHAPES.map((entry) => (
                <button
                  type="button"
                  key={entry.id}
                  className={`sw-chip ${shapeId === entry.id ? "on" : ""}`.trim()}
                  aria-pressed={shapeId === entry.id}
                  onClick={() => setShapeId(entry.id)}
                >
                  {entry.label}
                </button>
              ))}
            </div>
          </div>

          <div className="sw-go">
            <button type="button" className="sw-cta" onClick={handleCreate} disabled={!canSubmit}>
              <Icon.Wand />
              <span>{submitting ? "Starting…" : "Create"}</span>
            </button>
            <span className="sw-meta">
              Makes {count} {count === 1 ? "option" : "options"} · about a minute
            </span>
          </div>

          {notice ? <p className="sw-notice">{notice}</p> : null}

          <details className="sw-disclosure">
            <summary>
              <Icon.ChevDown className="sw-caret" /> More options
            </summary>
            <div className="sw-adv">
              <label>
                How many
                <select value={count} onChange={(event) => setCount(Number(event.target.value))}>
                  {COUNT_OPTIONS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Make it sharper
                <select value={upscaleId} onChange={(event) => setUpscaleId(event.target.value)}>
                  {UPSCALE_OPTIONS.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </details>
        </div>

        <div className="sw-results">
          <h3 className="sw-q">Latest</h3>
          {rendering ? <p className="sw-rendering">Rendering {imageLocalJobs.length} in progress…</p> : null}
          {recentImageAssets.length === 0 && !rendering ? (
            <div className="sw-empty">Nothing yet — describe an idea and press Create.</div>
          ) : (
            <div className="sw-grid">
              {recentImageAssets.slice(0, 8).map((asset) => (
                <button
                  type="button"
                  key={asset.id}
                  className="sw-shot"
                  onClick={() => setPreviewAsset?.(asset)}
                >
                  <AssetThumbnail asset={asset} />
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
