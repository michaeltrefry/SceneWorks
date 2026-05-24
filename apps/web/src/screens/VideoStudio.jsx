import React, { useEffect, useRef, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia, assetCanRenderAsVideo } from "../components/assetMedia.jsx";
import { Icon } from "../components/Icons.jsx";
import { JobProgressCard } from "../components/JobProgress.jsx";

const MOTIONS = [
  "static",
  "slow push-in",
  "pull out",
  "pan left",
  "pan right",
  "tilt up",
  "tilt down",
  "handheld",
];

function formatGpuLabel(requestedGpu) {
  if (!requestedGpu || requestedGpu === "auto") {
    return "auto GPU";
  }
  return `GPU ${requestedGpu}`;
}

function estimateRenderSeconds(durationSeconds, quality) {
  // Rough heuristic: every clip second ~3s on Balanced, ±50% for Draft/Final.
  const base = Math.max(1, Number(durationSeconds) || 6) * 3;
  if (quality === "fast") return Math.round(base * 0.5);
  if (quality === "best") return Math.round(base * 1.5);
  return Math.round(base);
}

function formatPlaybackTime(seconds) {
  const safeSeconds = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(safeSeconds / 60);
  return `${minutes}:${String(safeSeconds % 60).padStart(2, "0")}`;
}
import {
  clearPresetDefault,
  loraLooksLikeIcLora,
  noPresetId,
  rememberPresetDefault,
} from "../presetUtils.js";
import {
  onPromptKeyDown,
  PresetGuidanceStrip,
  PresetValidationWarnings,
  useGenerationStudio,
} from "./generationStudio.jsx";
import { ReplacePersonPanel, findReplacementModel } from "./ReplacePersonPanel.jsx";

const ltxVideoModelId = "ltx_2_3";
const ltxIcLoraRequiredModes = new Set(["extend_clip", "video_bridge"]);

export function VideoStudio({
  activeProject,
  assets,
  characters,
  createPersonDetectionJob,
  createPersonTrackJob,
  createVideoJob,
  deleteAsset,
  purgeAsset,
  gpuOptions,
  latestAssets,
  launchRequest,
  loras = [],
  jobs = [],
  localJobs: trackedLocalJobs = [],
  onCancelJob,
  onLocalJobCreated,
  onOpenPresets,
  onOpenQueue,
  onPreview,
  onSendToEditor,
  personTracks = [],
  personReadiness = {},
  presets = [],
  requestedGpu,
  saveTrackCorrections,
  selectedAsset,
  setRequestedGpu,
  updateAssetStatus,
  videoModels,
}) {
  const [motion, setMotion] = useState("slow push-in");
  const imageAssets = assets.filter((asset) => asset.type === "image" || asset.type === "frame");
  const videoAssets = assets.filter((asset) => asset.type === "video");
  const [mode, setMode] = useState("image_to_video");
  const [prompt, setPrompt] = useState("Camera slowly pushes in while the scene comes alive");
  const [quality, setQuality] = useState("balanced");
  const [ltxPipeline, setLtxPipeline] = useState("auto");
  const [distilledVariant, setDistilledVariant] = useState("1.1");
  const [precision, setPrecision] = useState("fp8");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [model, setModel] = useState(videoModels[0]?.id ?? ltxVideoModelId);
  const selectedModel = videoModels.find((item) => item.id === model) ?? videoModels[0];
  const [duration, setDuration] = useState(selectedModel?.defaults?.duration ?? 6);
  const [resolution, setResolution] = useState(selectedModel?.defaults?.resolution ?? "768x512");
  const [fps, setFps] = useState(selectedModel?.defaults?.fps ?? 25);
  const [seed, setSeed] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [sourceAssetId, setSourceAssetId] = useState(["image", "frame"].includes(selectedAsset?.type) ? selectedAsset.id : "");
  const [lastFrameAssetId, setLastFrameAssetId] = useState("");
  const [sourceClipAssetId, setSourceClipAssetId] = useState(selectedAsset?.type === "video" ? selectedAsset.id : "");
  const [characterId, setCharacterId] = useState("");
  const [characterLookId, setCharacterLookId] = useState("");
  const [personTrackId, setPersonTrackId] = useState("");
  const [replacementMode, setReplacementMode] = useState("face_only");
  const [selectedDetectionId, setSelectedDetectionId] = useState("");
  const [trackName, setTrackName] = useState("Selected person");
  const [comparisonMode, setComparisonMode] = useState("side_by_side");
  const [abSide, setAbSide] = useState("replacement");
  const [submitting, setSubmitting] = useState(false);
  const previewVideoRef = useRef(null);
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewTime, setPreviewTime] = useState(0);
  const [previewDuration, setPreviewDuration] = useState(0);
  const presetDefaultSnapshots = useRef({});
  const capabilities = selectedModel?.capabilities ?? [];
  const supportsMode = capabilities.includes(mode);
  const implementedMode = ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "replace_person"].includes(mode);
  const {
    availablePresets,
    selectedPreset,
    setSelectedPresetId,
    presetPromptParts,
    presetLoraDetails,
    presetValidationResult,
    localJobs,
  } = useGenerationStudio({
    mode,
    presets,
    selectedModel,
    loras,
    models: videoModels,
    model,
    setModel,
    fallbackModelId: ltxVideoModelId,
    characters,
    characterId,
    setCharacterId,
    setCharacterLookId,
    assets,
    latestAssets,
    trackedLocalJobs,
  });
  const requiresLtxIcLora = selectedModel?.id === ltxVideoModelId && ltxIcLoraRequiredModes.has(mode);
  const hasLtxIcLora = presetLoraDetails.some((lora) => !lora.missing && loraLooksLikeIcLora(lora));

  useEffect(() => {
    if (selectedAsset?.type === "image" || selectedAsset?.type === "frame") {
      setSourceAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
  }, [selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (launchRequest?.view !== "Video") {
      return;
    }
    if (launchRequest.characterId) {
      setMode(launchRequest.mode ?? "text_to_video");
      setCharacterId(launchRequest.characterId);
      setCharacterLookId(launchRequest.lookId ?? "");
      return;
    }
    if (launchRequest.assetId !== selectedAsset?.id) {
      return;
    }
    setMode(launchRequest.mode);
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "image" || selectedAsset?.type === "frame") {
      setSourceAssetId(selectedAsset.id);
    }
  }, [launchRequest?.id, selectedAsset?.id, selectedAsset?.type]);

  useEffect(() => {
    if (!selectedModel) {
      return;
    }
    setDuration((current) => {
      const options = selectedModel.limits?.durations ?? [4, 6, 8, 10];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.duration ?? options[0];
    });
    setResolution((current) => {
      const options = selectedModel.limits?.resolutions ?? ["768x512"];
      return options.includes(current) ? current : selectedModel.defaults?.resolution ?? options[0];
    });
    setFps((current) => {
      const options = selectedModel.limits?.fps ?? [24, 25, 30];
      return options.includes(Number(current)) ? current : selectedModel.defaults?.fps ?? options[0];
    });
  }, [selectedModel?.id]);

  useEffect(() => {
    if (mode !== "replace_person" || supportsMode) {
      return;
    }
    const replacementModel = findReplacementModel(videoModels);
    if (replacementModel) {
      setModel(replacementModel.id);
    }
  }, [mode, supportsMode, videoModels]);

  useEffect(() => {
    if (!selectedPreset) {
      clearPresetDefault(setDuration, presetDefaultSnapshots, "duration");
      clearPresetDefault(setFps, presetDefaultSnapshots, "fps");
      clearPresetDefault(setQuality, presetDefaultSnapshots, "quality");
      clearPresetDefault(setResolution, presetDefaultSnapshots, "resolution");
      clearPresetDefault(setNegativePrompt, presetDefaultSnapshots, "negativePrompt");
      return;
    }
    const defaults = selectedPreset.defaults ?? {};
    if (defaults.duration) {
      const appliedValue = Number(defaults.duration);
      setDuration((current) => {
        rememberPresetDefault(presetDefaultSnapshots, "duration", current, appliedValue);
        return appliedValue;
      });
    }
    if (defaults.fps) {
      const appliedValue = Number(defaults.fps);
      setFps((current) => {
        rememberPresetDefault(presetDefaultSnapshots, "fps", current, appliedValue);
        return appliedValue;
      });
    }
    if (defaults.quality) {
      const appliedValue = defaults.quality;
      setQuality((current) => {
        rememberPresetDefault(presetDefaultSnapshots, "quality", current, appliedValue);
        return appliedValue;
      });
    }
    if (defaults.resolution) {
      const appliedValue = defaults.resolution;
      setResolution((current) => {
        rememberPresetDefault(presetDefaultSnapshots, "resolution", current, appliedValue);
        return appliedValue;
      });
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "negativePrompt")) {
      const appliedValue = defaults.negativePrompt ?? "";
      setNegativePrompt((current) => {
        rememberPresetDefault(presetDefaultSnapshots, "negativePrompt", current, appliedValue);
        return appliedValue;
      });
    }
  }, [selectedPreset?.id]);

  useEffect(() => {
    if (mode !== "replace_person") {
      return;
    }
    const firstMatchingTrack = personTracks.find((track) => track.sourceAssetId === sourceClipAssetId);
    if (firstMatchingTrack && !personTracks.some((track) => track.id === personTrackId)) {
      setPersonTrackId(firstMatchingTrack.id);
    }
  }, [mode, personTracks, personTrackId, sourceClipAssetId]);

  const modeOptions = [
    ["image_to_video", "Image → Video"],
    ["text_to_video", "Text → Video"],
    ["first_last_frame", "First → Last"],
    ["extend_clip", "Extend"],
    ["replace_person", "Replace person"],
  ];
  const matchingTracks = personTracks.filter((track) => track.sourceAssetId === sourceClipAssetId);
  const latestDetectionJob = jobs
    .filter(
      (job) =>
        job.type === "person_detect" &&
        job.status === "completed" &&
        job.projectId === activeProject?.id &&
        job.payload?.sourceAssetId === sourceClipAssetId,
    )
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
  const detectionResult = latestDetectionJob?.result ?? null;
  const representativeFrame = assets.find((asset) => asset.id === detectionResult?.frameAssetId);
  const selectedDetection = detectionResult?.detections?.find((item) => item.id === selectedDetectionId) ?? detectionResult?.detections?.[0];
  const selectedTrack = personTracks.find((track) => track.id === personTrackId);
  const comparisonAsset = latestAssets.find((asset) => asset.recipe?.mode === "replace_person");
  const comparisonSource = assets.find((asset) => asset.id === comparisonAsset?.lineage?.sourceClipAssetId);
  const hasInputs =
    mode === "text_to_video" ||
    (mode === "image_to_video" && sourceAssetId) ||
    (mode === "first_last_frame" && sourceAssetId && lastFrameAssetId) ||
    (mode === "extend_clip" && sourceClipAssetId) ||
    (mode === "replace_person" && sourceClipAssetId && personTrackId && characterId);
  // Don't let Replace Person queue a job the readiness endpoint says no live
  // worker can run — that would sit unclaimable instead of honoring the gate.
  const replaceReady = mode !== "replace_person" || personReadiness?.replace?.ready !== false;
  const canSubmit = Boolean(
    activeProject &&
      prompt.trim() &&
      supportsMode &&
      implementedMode &&
      hasInputs &&
      presetValidationResult.ok &&
      (!requiresLtxIcLora || hasLtxIcLora) &&
      replaceReady,
  );
  const [width, height] = resolution.split("x").map((value) => Number(value));
  const durationOptions = selectedModel?.limits?.durations ?? [4, 6, 8, 10];
  const resolutionOptions = selectedModel?.limits?.resolutions ?? ["768x512", "640x640", "1280x720", "720x1280"];
  const fpsOptions = selectedModel?.limits?.fps ?? [24, 25, 30];
  const durationHint =
    selectedModel?.ui?.durationHint ??
    (selectedModel?.limits?.recommendedMaxDuration ? `Recommended: ${selectedModel.limits.recommendedMaxDuration}s or less.` : "");
  const blockedMessage = !supportsMode
    ? `${selectedModel?.name ?? "Selected model"} does not support this mode.`
    : !implementedMode
      ? "This entry point is reserved for the next runtime slice."
      : !hasInputs
        ? "Required inputs are missing."
        : requiresLtxIcLora && !hasLtxIcLora
          ? "LTX video-conditioned generation needs an installed IC-LoRA preset."
          : !replaceReady
            ? "No live GPU worker can run person replacement yet."
            : "";
  const replacementModeLabels = {
    face_only: "Face Only",
    full_person_keep_outfit: "Full Person, Keep Outfit",
    full_person_replace_outfit: "Full Person, Replace Outfit",
  };

  async function submit(event) {
    event.preventDefault();
    if (submitting) {
      return;
    }
    setSubmitting(true);
    try {
      const job = await createVideoJob({
        mode,
        prompt,
        negativePrompt,
        model,
        duration: Number(duration),
        fps: Number(fps),
        width,
        height,
        quality,
        seed: seed === "" ? null : Number(seed),
        recipePresetId: selectedPreset?.id ?? null,
        characterId: characterId || null,
        characterLookId: characterLookId || null,
        sourceAssetId: ["image_to_video", "first_last_frame"].includes(mode) ? sourceAssetId || null : null,
        lastFrameAssetId: mode === "first_last_frame" ? lastFrameAssetId || null : null,
        sourceClipAssetId: ["extend_clip", "replace_person"].includes(mode) ? sourceClipAssetId || null : null,
        personTrackId: mode === "replace_person" ? personTrackId || null : null,
        replacementMode: mode === "replace_person" ? replacementMode : "face_only",
        loras: presetLoraDetails.filter((lora) => !lora.missing),
        advanced: {
          resolution,
          durationHint,
          motion,
          recipePresetName: selectedPreset?.name ?? null,
          recipePresetPrompt: selectedPreset?.prompt ?? null,
          selectedPersonTrack: selectedTrack ?? null,
          replacementModeLabel: replacementModeLabels[replacementMode],
          ...(model === ltxVideoModelId ? { ltxPipeline, distilledVariant, precision } : {}),
        },
      });
      onLocalJobCreated?.(job);
    } finally {
      setSubmitting(false);
    }
  }

  const generateDisabled = submitting || !canSubmit;
  const renderLabel = mode === "replace_person" ? "Replace person" : "Render clip";
  const previewAsset = latestAssets[0] ?? null;
  const estimateSeconds = estimateRenderSeconds(duration, quality);
  const gpuLabel = formatGpuLabel(requestedGpu);
  const previewCanPlay = assetCanRenderAsVideo(previewAsset);
  const previewTotalSeconds = previewDuration || Number(previewAsset?.file?.duration) || Number(duration) || 0;
  const previewProgress = previewTotalSeconds ? `${Math.min(100, (previewTime / previewTotalSeconds) * 100)}%` : "0%";

  useEffect(() => {
    setPreviewPlaying(false);
    setPreviewTime(0);
    setPreviewDuration(0);
  }, [previewAsset?.id]);

  function togglePreviewPlayback() {
    const video = previewVideoRef.current;
    if (!video || !previewCanPlay) {
      return;
    }
    if (video.paused) {
      video.play().catch(() => setPreviewPlaying(false));
      return;
    }
    video.pause();
  }

  return (
    <section className="main-surface video-studio">
      <form className="studio-shell" onSubmit={submit}>
        <div className="surface-header hero studio-prompt-hero video-prompt-hero">
          <div className="prompt-hero-top">
            <div className="segmented-control mode-control" role="tablist" aria-label="Video mode">
              {modeOptions.map(([value, label]) => (
                <button
                  className={mode === value ? "active" : ""}
                  key={value}
                  onClick={() => setMode(value)}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
            {onOpenPresets ? (
              <button className="hero-link" onClick={onOpenPresets} type="button">
                <Icon.Folder size={14} /> Saved presets
              </button>
            ) : null}
          </div>

          <div className="prompt-input-row">
            <textarea
              aria-label="Prompt"
              className="prompt-input"
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={onPromptKeyDown}
              placeholder="Describe the motion — what moves, where the camera goes, how it feels…"
              value={prompt}
            />
            <button className="prompt-cta" disabled={generateDisabled} type="submit">
              <Icon.Sparkle size={14} />
              {submitting ? "Queueing…" : renderLabel}
            </button>
          </div>

          <div className="motion-row">
            <span className="motion-row-label">Motion:</span>
            {MOTIONS.map((option) => (
              <button
                className={motion === option ? "motion-chip active" : "motion-chip"}
                key={option}
                onClick={() => setMotion(option)}
                type="button"
              >
                <span aria-hidden="true" className="motion-arrow">→</span>
                {option}
              </button>
            ))}
          </div>
        </div>

        {mode !== "text_to_video" ? (
          <div className="studio-source-band">
            {mode === "image_to_video" || mode === "first_last_frame" ? (
              <AssetPickerField
                assets={imageAssets}
                buttonLabel="Select image"
                emptyLabel="No first frame selected"
                label="First frame"
                onChange={setSourceAssetId}
                value={sourceAssetId}
              />
            ) : null}

            {mode === "first_last_frame" ? (
              <AssetPickerField
                assets={imageAssets}
                buttonLabel="Select image"
                emptyLabel="No last frame selected"
                label="Last frame"
                onChange={setLastFrameAssetId}
                value={lastFrameAssetId}
              />
            ) : null}

            {mode === "extend_clip" ? (
              <AssetPickerField
                assets={videoAssets}
                buttonLabel="Select clip"
                emptyLabel="No source clip selected"
                label="Source clip"
                onChange={setSourceClipAssetId}
                value={sourceClipAssetId}
              />
            ) : null}

            {mode === "replace_person" ? (
              <ReplacePersonPanel
                createPersonDetectionJob={createPersonDetectionJob}
                createPersonTrackJob={createPersonTrackJob}
                personReadiness={personReadiness}
                detectionResult={detectionResult}
                matchingTracks={matchingTracks}
                personTrackId={personTrackId}
                replacementMode={replacementMode}
                representativeFrame={representativeFrame}
                saveTrackCorrections={saveTrackCorrections}
                selectedDetection={selectedDetection}
                selectedTrack={selectedTrack}
                setPersonTrackId={setPersonTrackId}
                setReplacementMode={setReplacementMode}
                setSelectedDetectionId={setSelectedDetectionId}
                setSourceClipAssetId={setSourceClipAssetId}
                setTrackName={setTrackName}
                sourceClipAssetId={sourceClipAssetId}
                trackName={trackName}
                videoAssets={videoAssets}
              />
            ) : null}
          </div>
        ) : null}

        <div className="video-results">
          <div className="video-main-stack">
            {localJobs.length ? (
              <div className="local-job-stack">
                {localJobs.map((job) => (
                  <JobProgressCard job={job} key={job.id} label="Video generation" onCancel={onCancelJob} onOpenQueue={onOpenQueue} />
                ))}
              </div>
            ) : null}

            <div className="video-preview-card">
              <div className="video-preview-stage">
                {previewAsset ? (
                  <AssetMedia
                    asset={previewAsset}
                    controls={false}
                    onEnded={() => setPreviewPlaying(false)}
                    onLoadedMetadata={(event) => setPreviewDuration(event.currentTarget.duration || 0)}
                    onPause={() => setPreviewPlaying(false)}
                    onPlay={() => setPreviewPlaying(true)}
                    onTimeUpdate={(event) => setPreviewTime(event.currentTarget.currentTime || 0)}
                    ref={previewVideoRef}
                  />
                ) : (
                  <span className="video-preview-empty">No clip rendered yet — set up the prompt above and hit Render</span>
                )}
              </div>

              <div className="video-playback-bar">
                <button
                  aria-label={previewPlaying ? "Pause preview" : "Play preview"}
                  className="play-btn"
                  disabled={!previewCanPlay}
                  onClick={togglePreviewPlayback}
                  type="button"
                >
                  {previewPlaying ? <Icon.Pause size={14} /> : <Icon.Play size={14} />}
                </button>
                <div aria-hidden="true" className="video-playback-scrub">
                  <span className="video-playback-scrub-fill" style={{ width: previewProgress }} />
                </div>
                <span className="video-playback-time">
                  {formatPlaybackTime(previewTime)} / {formatPlaybackTime(previewTotalSeconds)}
                </span>
                <span className="video-playback-estimate">~{estimateSeconds}s on {gpuLabel}</span>
                <button
                  className="send-editor-btn"
                  disabled={!previewAsset || !onSendToEditor}
                  onClick={() => previewAsset && onSendToEditor?.(previewAsset)}
                  type="button"
                >
                  <Icon.Editor size={14} /> Send to editor
                </button>
              </div>
            </div>

            {videoAssets.length ? (
              <div className="recent-clips-card">
                <div className="recent-clips-head">
                  <h3>Recent clips</h3>
                  <span className="meta">{localJobs.length || latestAssets.length} this session</span>
                </div>
                <div className="recent-clips-strip">
                  {videoAssets.slice(0, 4).map((asset) => (
                    <button className="tray-item" key={asset.id} onClick={() => onPreview(asset)} type="button">
                      <AssetMedia asset={asset} />
                      <span>{asset.displayName}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            {comparisonAsset?.recipe?.mode === "replace_person" && comparisonSource ? (
              <div className="comparison-panel">
                <div className="comparison-toolbar">
                  <div className="segmented-control compact-segment" aria-label="Comparison mode">
                    <button className={comparisonMode === "side_by_side" ? "active" : ""} onClick={() => setComparisonMode("side_by_side")} type="button">
                      Side by Side
                    </button>
                    <button className={comparisonMode === "ab" ? "active" : ""} onClick={() => setComparisonMode("ab")} type="button">
                      A/B
                    </button>
                  </div>
                  {comparisonMode === "ab" ? (
                    <div className="segmented-control compact-segment" aria-label="A/B source">
                      <button className={abSide === "original" ? "active" : ""} onClick={() => setAbSide("original")} type="button">
                        A
                      </button>
                      <button className={abSide === "replacement" ? "active" : ""} onClick={() => setAbSide("replacement")} type="button">
                        B
                      </button>
                    </div>
                  ) : null}
                </div>
                {comparisonMode === "side_by_side" ? (
                  <div className="comparison-grid">
                    <div>
                      <p className="eyebrow">Original</p>
                      <AssetMedia asset={comparisonSource} />
                    </div>
                    <div>
                      <p className="eyebrow">Replacement</p>
                      <AssetMedia asset={comparisonAsset} />
                    </div>
                  </div>
                ) : (
                  <div className="comparison-single">
                    <p className="eyebrow">{abSide === "original" ? "A Original" : "B Replacement"}</p>
                    <AssetMedia asset={abSide === "original" ? comparisonSource : comparisonAsset} />
                  </div>
                )}
              </div>
            ) : null}

            {latestAssets.length > 1 ? (
              <div className="review-grid video-review-grid">
                {latestAssets.slice(1).map((asset) => (
                  <AssetCard
                    asset={asset}
                    deleteAsset={deleteAsset}
                    key={asset.id}
                    onPreview={onPreview}
                    purgeAsset={purgeAsset}
                    updateAssetStatus={updateAssetStatus}
                  />
                ))}
              </div>
            ) : null}

            {blockedMessage ? <p className="inline-warning">{blockedMessage}</p> : null}
            <PresetValidationWarnings presetValidationResult={presetValidationResult} selectedModel={selectedModel} />
          </div>

          <div className="video-rail">
            <aside className="render-rail">
              <div className="preset-rail-head">
                <h3>Render settings</h3>
                <span className="preset-rail-model-tag">{selectedModel?.name ?? "—"}</span>
              </div>

              <label>
                Model
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {videoModels.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>

              {availablePresets.length ? (
                <div className="style-preset-strip">
                  <span className="style-preset-label">Style preset</span>
                  <div className="preset-chips">
                    <button
                      className={!selectedPreset ? "preset-chip active" : "preset-chip"}
                      onClick={() => setSelectedPresetId(noPresetId)}
                      type="button"
                    >
                      None
                    </button>
                    {availablePresets.map((preset) => (
                      <button
                        className={selectedPreset?.id === preset.id ? "preset-chip active" : "preset-chip"}
                        key={preset.id}
                        onClick={() => setSelectedPresetId(preset.id)}
                        type="button"
                      >
                        {preset.name ?? preset.id}
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}

              <label>
                Quality
                <div className="quality-segment" role="radiogroup" aria-label="Quality">
                  {[
                    ["fast", "Draft"],
                    ["balanced", "Balanced"],
                    ["best", "Final"],
                  ].map(([value, label]) => (
                    <button
                      aria-checked={quality === value}
                      className={quality === value ? "active" : ""}
                      key={value}
                      onClick={() => setQuality(value)}
                      role="radio"
                      type="button"
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </label>

              <div className="control-grid preset-rail-row">
                <label>
                  Duration
                  <select onChange={(event) => setDuration(Number(event.target.value))} value={duration}>
                    {durationOptions.map((value) => (
                      <option key={value} value={value}>
                        {value}s
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Frames
                  <select onChange={(event) => setFps(Number(event.target.value))} value={fps}>
                    {fpsOptions.map((value) => (
                      <option key={value} value={value}>
                        {value} fps
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <label>
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((value) => (
                    <option key={value} value={value}>
                      {value.replace("x", " × ")}
                    </option>
                  ))}
                </select>
              </label>

              <PresetGuidanceStrip
                selectedPreset={selectedPreset}
                presetPromptParts={presetPromptParts}
                presetLoraDetails={presetLoraDetails}
                noPresetHint="Generation uses only the prompt, model, and visible render settings."
              />

              {durationHint ? <p className="helper-copy">{durationHint}</p> : null}

              <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
                <Icon.ChevDown className={advancedOpen ? "chev-rotate open" : "chev-rotate"} size={14} />
                {advancedOpen ? "Hide advanced" : "Advanced"}
              </button>

              {advancedOpen ? (
                <div className="advanced-panel">
                  {model === ltxVideoModelId ? (
                    <>
                      <label>
                        LTX pipeline
                        <select onChange={(event) => setLtxPipeline(event.target.value)} value={ltxPipeline}>
                          <option value="auto">Auto (follow quality)</option>
                          <option value="distilled">Distilled (single-stage)</option>
                          <option value="two_stage">Two-stage (dev + upscaler)</option>
                        </select>
                      </label>
                      <label>
                        Distilled variant
                        <select onChange={(event) => setDistilledVariant(event.target.value)} value={distilledVariant}>
                          <option value="1.1">1.1 (newer aesthetic + audio)</option>
                          <option value="1.0">1.0 (original)</option>
                        </select>
                      </label>
                      <label>
                        Precision
                        <select onChange={(event) => setPrecision(event.target.value)} value={precision}>
                          <option value="fp8">FP8 (lower VRAM)</option>
                          <option value="bf16">BF16 (higher quality, CPU offload)</option>
                        </select>
                      </label>
                    </>
                  ) : null}
                  <label>
                    GPU
                    <select onChange={(event) => setRequestedGpu(event.target.value)} value={requestedGpu}>
                      {gpuOptions.map((gpu) => (
                        <option key={gpu} value={gpu}>
                          {gpu === "auto" ? "Auto" : gpu}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Seed
                    <input onChange={(event) => setSeed(event.target.value)} placeholder="Random" type="number" value={seed} />
                  </label>
                  <label>
                    Character
                    <select onChange={(event) => setCharacterId(event.target.value)} value={characterId}>
                      <option value="">No character</option>
                      {characters.map((character) => (
                        <option key={character.id} value={character.id}>
                          {character.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Look
                    <select onChange={(event) => setCharacterLookId(event.target.value)} value={characterLookId}>
                      <option value="">Default look</option>
                      {(characters.find((character) => character.id === characterId)?.looks ?? []).map((look) => (
                        <option key={look.id} value={look.id}>
                          {look.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="prompt-field">
                    Negative prompt
                    <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
                  </label>
                  {characterId ? (
                    <div className="guidance-strip">
                      <strong>Character reference</strong>
                      <span>
                        Character and look are saved with the recipe; LTX image conditioning uses IC-LoRA when the selected preset includes one.
                      </span>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </aside>

            <aside className="tips-card">
              <h3>Tips</h3>
              <ul>
                <li>Short clips (4–6s) compose better in the editor.</li>
                <li>Describe the motion, not just the scene.</li>
                <li>Pick a motion chip above to guide the camera.</li>
              </ul>
            </aside>

            <aside className="keyboard-card">
              <h3>Keyboard</h3>
              <dl>
                <div className="kbd-row">
                  <span>Render</span>
                  <span className="kbd-keys">
                    <kbd>⌘</kbd>
                    <kbd>↵</kbd>
                  </span>
                </div>
                <div className="kbd-row">
                  <span>Send to editor</span>
                  <span className="kbd-keys">
                    <kbd>⇧</kbd>
                    <kbd>E</kbd>
                  </span>
                </div>
                <div className="kbd-row">
                  <span>Loop preview</span>
                  <span className="kbd-keys">
                    <kbd>L</kbd>
                  </span>
                </div>
              </dl>
            </aside>
          </div>
        </div>
      </form>
    </section>
  );
}
