import React, { useEffect, useMemo, useRef, useState } from "react";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { JobProgressCard } from "../components/JobProgress.jsx";
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
  onOpenQueue,
  onPreview,
  personTracks = [],
  recipePresets = [],
  requestedGpu,
  selectedAsset,
  setRequestedGpu,
  updateAssetStatus,
  videoModels,
}) {
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
    ["image_to_video", "Image to Video"],
    ["text_to_video", "Text to Video"],
    ["first_last_frame", "First/Last Frame"],
    ["extend_clip", "Extend Clip"],
    ["replace_person", "Replace Person"],
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

  return (
    <section className="main-surface video-studio">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Video Studio</p>
          <h2>{activeProject ? activeProject.name : "Create a project"}</h2>
        </div>
        <div className="segmented-control mode-control" role="tablist" aria-label="Video mode">
          {modeOptions.map(([value, label]) => (
            <button className={mode === value ? "active" : ""} key={value} onClick={() => setMode(value)} type="button">
              {label}
            </button>
          ))}
        </div>
      </div>

      <form className="studio-layout video-layout" onSubmit={submit}>
        <section className="studio-controls">
          <div className="control-grid generation-primary-grid">
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
            <label>
              Preset
              <select onChange={(event) => setRecipePresetId(event.target.value)} value={selectedRecipePreset?.id ?? noRecipePresetId}>
                <option value={noRecipePresetId}>None</option>
                {availableRecipePresets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.name ?? preset.id}
                  </option>
                ))}
              </select>
            </label>
          </div>

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

          <div className="control-grid compact-controls">
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
          </div>
          {characterId ? (
            <div className="guidance-strip">
              <strong>Recipe-only character</strong>
              <span>Character and look are saved with the recipe; adapter-level reference and LoRA conditioning are not active yet.</span>
            </div>
          ) : null}

          <label className="prompt-field">
            Prompt
            <textarea onChange={(event) => setPrompt(event.target.value)} value={prompt} />
          </label>

          <div className="control-grid video-preset-grid">
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
              Quality
              <select onChange={(event) => setQuality(event.target.value)} value={quality}>
                <option value="fast">Fast</option>
                <option value="balanced">Balanced</option>
                <option value="best">Best</option>
              </select>
            </label>
          </div>

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
              <span>Generation will use only the visible prompt, model, and advanced settings.</span>
            </div>
          )}

          <div className="guidance-strip">
            <strong>{selectedModel?.name ?? "Video model"}</strong>
            <span>{durationHint}</span>
          </div>

          <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
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
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  {resolutionOptions.map((value) => (
                    <option key={value} value={value}>
                      {value.replace("x", " x ")}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                FPS
                <select onChange={(event) => setFps(Number(event.target.value))} value={fps}>
                  {fpsOptions.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Seed
                <input onChange={(event) => setSeed(event.target.value)} placeholder="Random" type="number" value={seed} />
              </label>
              <label className="prompt-field">
                Negative prompt
                <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
              </label>
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
          <button className="primary-action" disabled={submitting || !canSubmit} type="submit">
            {submitting ? "Queueing..." : mode === "replace_person" ? "Replace Person" : "Generate Clip"}
          </button>
        </section>

        <section className="review-panel video-review">
          <div className="section-heading">
            <p className="eyebrow">Fresh clip</p>
            <h2>Review</h2>
          </div>
          {localJobs.length ? (
            <div className="local-job-stack">
              {localJobs.map((job) => (
                <JobProgressCard job={job} key={job.id} label="Video generation" onOpenQueue={onOpenQueue} />
              ))}
            </div>
          ) : null}
          {latestAssets.length ? (
            <div className="review-grid video-review-grid">
              {latestAssets.map((asset) => (
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
          ) : hasReviewContent ? null : (
            <div className="empty-panel">No fresh video clip</div>
          )}

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

          <div className="asset-tray">
            <div className="section-heading">
              <p className="eyebrow">Asset tray</p>
              <h2>Recent videos</h2>
            </div>
            <div className="tray-grid">
              {videoAssets.slice(0, 4).map((asset) => (
                <button className="tray-item" key={asset.id} onClick={() => onPreview(asset)} type="button">
                  <AssetMedia asset={asset} />
                  <span>{asset.displayName}</span>
                </button>
              ))}
              {videoAssets.length === 0 ? <div className="empty-panel compact-panel">No video assets</div> : null}
            </div>
          </div>
        </section>
      </form>
    </section>
  );
}
