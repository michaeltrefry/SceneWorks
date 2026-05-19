import React, { useEffect, useMemo, useRef, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
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
import {
  clearPresetDefault,
  noRecipePresetId,
  presetLoraDetails as buildPresetLoraDetails,
  presetMatchesModel,
  presetMatchesWorkflow,
  presetPromptParts as buildPresetPromptParts,
  presetValidation,
  rememberPresetDefault,
} from "../presetUtils.js";
import { ReplacePersonPanel, findReplacementModel } from "./ReplacePersonPanel.jsx";

const completedResultFallbackMs = 30000;

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
  onLocalJobCreated,
  onOpenPresets,
  onOpenQueue,
  onPreview,
  onSendToEditor,
  personTracks = [],
  recipePresets = [],
  requestedGpu,
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
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [model, setModel] = useState(videoModels[0]?.id ?? "ltx_2_3");
  const [recipePresetId, setRecipePresetId] = useState(null);
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
  const [resultFallbackTick, setResultFallbackTick] = useState(0);
  const presetDefaultSnapshots = useRef({});
  const capabilities = selectedModel?.capabilities ?? [];
  const supportsMode = capabilities.includes(mode);
  const implementedMode = ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "replace_person"].includes(mode);
  const availableRecipePresets = useMemo(() => {
    return recipePresets.filter((preset) => presetMatchesWorkflow(preset, mode) && presetMatchesModel(preset, selectedModel));
  }, [mode, recipePresets, selectedModel?.id]);
  const selectedRecipePreset =
    recipePresetId === noRecipePresetId
      ? null
      : recipePresetId
        ? availableRecipePresets.find((preset) => preset.id === recipePresetId) ?? null
        : availableRecipePresets[0] ?? null;
  const presetPromptParts = buildPresetPromptParts(selectedRecipePreset);
  const presetLoraDetails = buildPresetLoraDetails(selectedRecipePreset, loras);
  const presetValidationResult = useMemo(
    () => presetValidation(selectedRecipePreset, loras, selectedModel),
    [selectedRecipePreset, loras, selectedModel],
  );

  useEffect(() => {
    if (!videoModels.some((item) => item.id === model)) {
      setModel(videoModels[0]?.id ?? "ltx_2_3");
    }
  }, [videoModels, model]);

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
    if (characterId && !characters.some((character) => character.id === characterId)) {
      setCharacterId("");
      setCharacterLookId("");
    }
  }, [characters, characterId]);

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
    if (!recipePresetId || recipePresetId === noRecipePresetId) {
      return;
    }
    if (!selectedRecipePreset) {
      setRecipePresetId(availableRecipePresets[0]?.id ?? noRecipePresetId);
    }
  }, [availableRecipePresets, recipePresetId, selectedRecipePreset]);

  useEffect(() => {
    if (!selectedRecipePreset) {
      clearPresetDefault(setDuration, presetDefaultSnapshots, "duration");
      clearPresetDefault(setFps, presetDefaultSnapshots, "fps");
      clearPresetDefault(setQuality, presetDefaultSnapshots, "quality");
      clearPresetDefault(setResolution, presetDefaultSnapshots, "resolution");
      clearPresetDefault(setNegativePrompt, presetDefaultSnapshots, "negativePrompt");
      return;
    }
    const defaults = selectedRecipePreset.defaults ?? {};
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
  }, [selectedRecipePreset?.id]);

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
  const canSubmit = Boolean(activeProject && prompt.trim() && supportsMode && implementedMode && hasInputs && presetValidationResult.ok);
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
        : "";
  const replacementModeLabels = {
    face_only: "Face Only",
    full_person_keep_outfit: "Full Person, Keep Outfit",
    full_person_replace_outfit: "Full Person, Replace Outfit",
  };

  function resultVisible(job) {
    if (job.result?.generationSetId) {
      return latestAssets.some((asset) => asset.generationSetId === job.result.generationSetId);
    }
    const assetIds = job.result?.assetIds ?? [];
    return assetIds.length > 0 && assetIds.every((id) => assets.some((asset) => asset.id === id));
  }

  function completedAnchorMs(job) {
    return Date.parse(job.completedAt ?? job.updatedAt ?? "");
  }

  function completedWaitExpired(job, nowMs = Date.now()) {
    const anchorMs = completedAnchorMs(job);
    return Number.isFinite(anchorMs) && nowMs - anchorMs > completedResultFallbackMs;
  }

  useEffect(() => {
    const nowMs = Date.now();
    const pendingCompletedJobs = trackedLocalJobs.filter(
      (job) =>
        job.status === "completed" &&
        Number.isFinite(completedAnchorMs(job)) &&
        !resultVisible(job) &&
        !completedWaitExpired(job, nowMs),
    );
    if (!pendingCompletedJobs.length) {
      return undefined;
    }
    const nextDelay = Math.min(
      ...pendingCompletedJobs.map((job) => Math.max(0, completedResultFallbackMs - (nowMs - completedAnchorMs(job)))),
    );
    const timer = window.setTimeout(() => setResultFallbackTick((value) => value + 1), nextDelay + 50);
    return () => window.clearTimeout(timer);
  }, [assets, latestAssets, trackedLocalJobs, resultFallbackTick]);

  const localJobs = trackedLocalJobs.filter(
    (job) => job.status !== "completed" || (!resultVisible(job) && !completedWaitExpired(job)),
  );
  const hasReviewContent = Boolean(localJobs.length || latestAssets.length);

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
        recipePresetId: selectedRecipePreset?.id ?? null,
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
          recipePresetName: selectedRecipePreset?.name ?? null,
          recipePresetPrompt: selectedRecipePreset?.prompt ?? null,
          selectedPersonTrack: selectedTrack ?? null,
          replacementModeLabel: replacementModeLabels[replacementMode],
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

  function onPromptKeyDown(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
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
                <Icon.Folder size={14} /> Saved recipes
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
                detectionResult={detectionResult}
                matchingTracks={matchingTracks}
                personTrackId={personTrackId}
                replacementMode={replacementMode}
                representativeFrame={representativeFrame}
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
                  <JobProgressCard job={job} key={job.id} label="Video generation" onOpenQueue={onOpenQueue} />
                ))}
              </div>
            ) : null}

            <div className="video-preview-card">
              <div className="video-preview-stage">
                {previewAsset ? (
                  <AssetMedia asset={previewAsset} />
                ) : (
                  <span className="video-preview-empty">No clip rendered yet — set up the prompt above and hit Render</span>
                )}
              </div>

              <div className="video-playback-bar">
                <button aria-label="Play preview" className="play-btn" type="button">
                  <Icon.Play size={14} />
                </button>
                <div aria-hidden="true" className="video-playback-scrub">
                  <span className="video-playback-scrub-fill" />
                </div>
                <span className="video-playback-time">0:00 / 0:{String(Math.round(Number(duration) || 0)).padStart(2, "0")}</span>
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
            {presetValidationResult.missing.length ? (
              <p className="inline-warning">
                Preset cannot run until LoRA import finishes: {presetValidationResult.missing.join(", ")}. Wait for the Queue or choose another preset.
              </p>
            ) : null}
            {presetValidationResult.incompatible.length ? (
              <p className="inline-warning">
                Preset cannot run with {selectedModel?.name ?? "the selected model"} because these LoRAs are incompatible: {presetValidationResult.incompatible.join(", ")}. Choose another preset or model.
              </p>
            ) : null}
          </div>

          <div className="video-rail">
            <aside className="render-rail">
              <div className="recipe-head">
                <h3>Render settings</h3>
                <span className="recipe-model-tag">{selectedModel?.name ?? "—"}</span>
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

              {availableRecipePresets.length ? (
                <div className="style-preset-strip">
                  <span className="style-preset-label">Style preset</span>
                  <div className="preset-chips">
                    <button
                      className={!selectedRecipePreset ? "preset-chip active" : "preset-chip"}
                      onClick={() => setRecipePresetId(noRecipePresetId)}
                      type="button"
                    >
                      None
                    </button>
                    {availableRecipePresets.map((preset) => (
                      <button
                        className={selectedRecipePreset?.id === preset.id ? "preset-chip active" : "preset-chip"}
                        key={preset.id}
                        onClick={() => setRecipePresetId(preset.id)}
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

              <div className="control-grid recipe-row">
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

              {selectedRecipePreset ? (
                <div className="guidance-strip">
                  <strong>{selectedRecipePreset.ui?.description ?? "Preset defaults active"}</strong>
                  <span>
                    {presetPromptParts.length ? `Adds: ${presetPromptParts.join(", ")}` : "No prompt fragments"}
                    {presetLoraDetails.length
                      ? ` | Preset LoRA applied at generation: ${presetLoraDetails.map((lora) => lora.name ?? lora.id).join(", ")}`
                      : " | No preset LoRAs"}
                    {presetLoraDetails.some((lora) => lora.missing) ? " | Import still pending" : ""}
                  </span>
                </div>
              ) : (
                <div className="guidance-strip">
                  <strong>No preset selected</strong>
                  <span>Generation uses only the prompt, model, and visible render settings.</span>
                </div>
              )}

              {durationHint ? <p className="helper-copy">{durationHint}</p> : null}

              <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
                <Icon.ChevDown className={advancedOpen ? "chev-rotate open" : "chev-rotate"} size={14} />
                {advancedOpen ? "Hide advanced" : "Advanced"}
              </button>

              {advancedOpen ? (
                <div className="advanced-panel">
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
                      <strong>Recipe-only character</strong>
                      <span>Character and look are saved with the recipe; adapter-level reference and LoRA conditioning are not active yet.</span>
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
