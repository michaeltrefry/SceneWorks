import React, { useMemo, useState } from "react";
import { useAppContext } from "../../context/AppContext.js";
import { Icon } from "../../components/Icons.jsx";
import { AssetThumbnail } from "../../components/assetMedia.jsx";
import { AssetPickerField } from "../../components/AssetPicker.jsx";
import { effectiveFitMode } from "../../components/FitModeControl.jsx";
import { pickClosestResolution, parseResolution } from "../../resolutionMatch.js";
import { LookTile } from "./LookTile.jsx";
import { useLookExemplars } from "./useLookExemplars.js";
import { modelLabel, useSimpleVideoModel } from "./simpleModel.js";
import { readPref, writePref } from "./simplePrefs.js";
import { AdvField } from "./AdvField.jsx";
import {
  LOOKS,
  SHAPES,
  VIDEO_MOTIONS,
  QUALITY_CHOICES,
  FALLBACK_VIDEO_RESOLUTIONS,
  composePrompt,
  videoDurations,
} from "./simpleDefaults.js";

// Snap a friendly shape to the closest resolution the video model supports.
function resolveDims(model, shape) {
  const options = model?.limits?.resolutions?.length ? model.limits.resolutions : FALLBACK_VIDEO_RESOLUTIONS;
  const picked = pickClosestResolution(shape.width, shape.height, options) ?? model?.defaults?.resolution ?? options[0];
  const dims = parseResolution(picked) ?? { width: 768, height: 512 };
  return { ...dims, resolution: picked };
}

export function MakeVideo() {
  const {
    activeProject,
    createVideoJob,
    refinePrompt,
    mediaAssets = [],
    recentVideoAssets = [],
    videoLocalJobs = [],
    setPreviewAsset,
  } = useAppContext();

  const [startMode, setStartMode] = useState("text"); // "text" | "picture"
  const [prompt, setPrompt] = useState("");
  const [sourceAssetId, setSourceAssetId] = useState("");
  const [motionId, setMotionId] = useState("push");
  const [lookId, setLookId] = useState(null);
  const [shapeId, setShapeId] = useState(() => readPref("video-shape") || "wide");
  const [durationValue, setDurationValue] = useState(null);
  const [quality, setQuality] = useState(() => readPref("video-quality") || "balanced");
  const [submitting, setSubmitting] = useState(false);
  const [describing, setDescribing] = useState(false);
  const [notice, setNotice] = useState("");

  // "Make my default" persistence for the video dropdowns (the model has its own).
  const [savedDefaults, setSavedDefaults] = useState(() => ({
    "video-quality": readPref("video-quality"),
    "video-shape": readPref("video-shape"),
  }));
  const saveDefault = (key, value) => {
    writePref(key, value);
    setSavedDefaults((current) => ({ ...current, [key]: String(value) }));
  };
  const isFieldDefault = (key, value) => savedDefaults[key] === String(value);

  const looks = useLookExemplars();
  const {
    models: videoChoices,
    modelId,
    select: selectModel,
    makeDefault,
    isDefaultId,
  } = useSimpleVideoModel();
  const motion = useMemo(() => VIDEO_MOTIONS.find((entry) => entry.id === motionId) ?? VIDEO_MOTIONS[0], [motionId]);
  const look = useMemo(() => LOOKS.find((entry) => entry.id === lookId) ?? null, [lookId]);
  const shape = useMemo(() => SHAPES.find((entry) => entry.id === shapeId) ?? SHAPES[0], [shapeId]);
  const requiredCapability = startMode === "picture" ? "image_to_video" : "text_to_video";
  const modeVideoChoices = useMemo(
    () => videoChoices.filter((entry) => (entry.capabilities ?? []).includes(requiredCapability)),
    [videoChoices, requiredCapability],
  );
  const modeModel = useMemo(
    () => modeVideoChoices.find((entry) => entry.id === modelId) ?? modeVideoChoices[0] ?? null,
    [modeVideoChoices, modelId],
  );
  const modeModelId = modeModel?.id ?? null;

  // Length options come from the selected model (lengths vary per model), capped
  // at the model's recommended max for Simple. Clamp the chosen value to those
  // options so switching models can't leave an unsupported/over-long length set.
  const durations = videoDurations(modeModel);
  const preferredDuration = durationValue ?? modeModel?.defaults?.duration;
  const duration = durations.some((value) => Number(value) === Number(preferredDuration))
    ? preferredDuration
    : durations[durations.length - 1];
  const fps = modeModel?.defaults?.fps ?? modeModel?.limits?.fps?.[0] ?? 25;

  // Only stills make sense as a first frame.
  const startImages = useMemo(
    () => mediaAssets.filter((asset) => ["image", "frame", "upload", "render"].includes(asset.type)),
    [mediaAssets],
  );

  const rendering = videoLocalJobs.length > 0;
  const needsSource = startMode === "picture" && !sourceAssetId;
  const modelNotice = !modeModelId
    ? startMode === "picture"
      ? "Add a video model that can animate pictures in Settings first."
      : "Add a text-to-video model in Settings first."
    : "";
  const canSubmit = Boolean(activeProject) && Boolean(modeModelId) && prompt.trim().length > 0 && !needsSource && !submitting;

  async function handleCreate() {
    if (!canSubmit) {
      if (!activeProject) setNotice("Open or create a workspace first.");
      else if (!modeModelId) setNotice(modelNotice);
      else if (needsSource) setNotice("Pick a picture to animate, or switch to “A description”.");
      return;
    }
    setSubmitting(true);
    setNotice("");
    const dims = resolveDims(modeModel, shape);
    const isImage = startMode === "picture";
    try {
      const job = await createVideoJob({
        mode: isImage ? "image_to_video" : "text_to_video",
        prompt: composePrompt(prompt, look),
        negativePrompt: "",
        model: modeModelId,
        duration: Number(duration),
        fps: Number(fps),
        width: dims.width,
        height: dims.height,
        quality,
        seed: null,
        recipePresetId: look?.presetId ?? null,
        characterId: null,
        characterLookId: null,
        sourceAssetId: isImage ? sourceAssetId || null : null,
        fitMode: isImage ? effectiveFitMode("crop", false) : undefined,
        lastFrameAssetId: null,
        loras: [],
        advanced: { resolution: dims.resolution, motion: motion.motion },
      });
      if (!job) setNotice("Couldn't start that video — check the workspace and try again.");
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
      const refined = await refinePrompt({ prompt: base, modelId: modeModelId, workflow: "video" });
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
          <h3 className="sw-q">Start from…</h3>
          <div className="sw-startfrom">
            <button
              type="button"
              className={`sw-sf ${startMode === "text" ? "on" : ""}`.trim()}
              aria-pressed={startMode === "text"}
              onClick={() => setStartMode("text")}
            >
              <b><Icon.Sparkle /> A description</b>
              <span>Type an idea and we'll film it.</span>
            </button>
            <button
              type="button"
              className={`sw-sf ${startMode === "picture" ? "on" : ""}`.trim()}
              aria-pressed={startMode === "picture"}
              onClick={() => setStartMode("picture")}
            >
              <b><Icon.Image /> A picture</b>
              <span>Animate a still you already have.</span>
            </button>
          </div>

          {startMode === "picture" ? (
            <div className="sw-field sw-pickwrap">
              <AssetPickerField
                assets={startImages}
                label="Picture to animate"
                buttonLabel="Choose a picture"
                emptyLabel="No picture chosen yet"
                value={sourceAssetId}
                onChange={setSourceAssetId}
                showCategories={false}
              />
            </div>
          ) : null}

          <div className="sw-field">
            <h3 className="sw-q">What happens in the video?</h3>
            <textarea
              className="sw-prompt"
              rows={3}
              value={prompt}
              placeholder="Snow drifting past the cabin windows as the lights flicker warmly"
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
          </div>

          <div className="sw-field">
            <h3 className="sw-q">Movement</h3>
            <div className="sw-chips">
              {VIDEO_MOTIONS.map((entry) => (
                <button
                  type="button"
                  key={entry.id}
                  className={`sw-chip ${motionId === entry.id ? "on" : ""}`.trim()}
                  aria-pressed={motionId === entry.id}
                  onClick={() => setMotionId(entry.id)}
                >
                  {entry.label}
                </button>
              ))}
            </div>
          </div>

          <div className="sw-field">
            <h3 className="sw-q">Length</h3>
            <div className="sw-chips">
              {durations.map((value) => (
                <button
                  type="button"
                  key={value}
                  className={`sw-chip ${Number(duration) === Number(value) ? "on" : ""}`.trim()}
                  aria-pressed={Number(duration) === Number(value)}
                  onClick={() => setDurationValue(value)}
                >
                  {value} seconds
                </button>
              ))}
            </div>
          </div>

          <div className="sw-field">
            <h3 className="sw-q sw-look-head">
              <span>Look <span className="sw-opt">— optional</span></span>
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

          <div className="sw-go">
            <button type="button" className="sw-cta" onClick={handleCreate} disabled={!canSubmit}>
              <Icon.Video />
              <span>{submitting ? "Starting…" : "Create video"}</span>
            </button>
            <span className="sw-meta">Makes 1 clip · takes a few minutes</span>
          </div>

          {notice || modelNotice ? <p className="sw-notice">{notice || modelNotice}</p> : null}

          <details className="sw-disclosure">
            <summary>
              <Icon.ChevDown className="sw-caret" /> More options
            </summary>
            <div className="sw-adv">
              {modeVideoChoices.length > 0 ? (
                <AdvField
                  label="Model"
                  isDefault={isDefaultId(modeModelId)}
                  onMakeDefault={() => makeDefault(modeModelId)}
                >
                  <select value={modeModelId ?? ""} onChange={(event) => selectModel(event.target.value)}>
                    {modeVideoChoices.map((entry) => (
                      <option key={entry.id} value={entry.id}>
                        {modelLabel(entry)}
                      </option>
                    ))}
                  </select>
                </AdvField>
              ) : null}
              <AdvField label="Quality" isDefault={isFieldDefault("video-quality", quality)} onMakeDefault={() => saveDefault("video-quality", quality)}>
                <select value={quality} onChange={(event) => setQuality(event.target.value)}>
                  {QUALITY_CHOICES.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.label}
                    </option>
                  ))}
                </select>
              </AdvField>
              <AdvField label="Shape" isDefault={isFieldDefault("video-shape", shapeId)} onMakeDefault={() => saveDefault("video-shape", shapeId)}>
                <select value={shapeId} onChange={(event) => setShapeId(event.target.value)}>
                  {SHAPES.map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.label}
                    </option>
                  ))}
                </select>
              </AdvField>
            </div>
          </details>
        </div>

        <div className="sw-results">
          <h3 className="sw-q">Latest clip</h3>
          {rendering ? <p className="sw-rendering">Rendering {videoLocalJobs.length} in progress…</p> : null}
          {recentVideoAssets.length === 0 && !rendering ? (
            <div className="sw-empty">Nothing yet — describe a video and press Create.</div>
          ) : (
            <div className="sw-grid sw-grid-video">
              {recentVideoAssets.slice(0, 6).map((asset) => (
                <button type="button" key={asset.id} className="sw-shot" onClick={() => setPreviewAsset?.(asset)}>
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
