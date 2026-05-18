import React, { useEffect, useMemo, useState } from "react";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";
import { JobProgressCard } from "../components/JobProgress.jsx";
import {
  presetLoraDetails as buildPresetLoraDetails,
  presetMatchesModel,
  presetMatchesWorkflow,
  presetPromptParts as buildPresetPromptParts,
} from "../presetUtils.js";
import { ReplacePersonPanel, findReplacementModel } from "./ReplacePersonPanel.jsx";

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
  const [recipePresetId, setRecipePresetId] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [model, setModel] = useState(videoModels[0]?.id ?? "ltx_2_3");
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
  const [localJobIds, setLocalJobIds] = useState([]);
  const capabilities = selectedModel?.capabilities ?? [];
  const supportsMode = capabilities.includes(mode);
  const implementedMode = ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "replace_person"].includes(mode);
  const availableRecipePresets = useMemo(() => {
    return recipePresets.filter((preset) => presetMatchesWorkflow(preset, mode) && presetMatchesModel(preset, selectedModel));
  }, [mode, recipePresets, selectedModel?.id]);
  const selectedRecipePreset = availableRecipePresets.find((preset) => preset.id === recipePresetId) ?? availableRecipePresets[0] ?? null;
  const presetPromptParts = buildPresetPromptParts(selectedRecipePreset);
  const presetLoraDetails = buildPresetLoraDetails(selectedRecipePreset, loras);
  const presetMissingLoras = presetLoraDetails.filter((lora) => lora.missing);

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
    if (!availableRecipePresets.some((preset) => preset.id === recipePresetId)) {
      setRecipePresetId(availableRecipePresets[0]?.id ?? "");
    }
  }, [availableRecipePresets, recipePresetId]);

  useEffect(() => {
    if (!selectedRecipePreset) {
      return;
    }
    const defaults = selectedRecipePreset.defaults ?? {};
    if (defaults.duration) {
      setDuration(Number(defaults.duration));
    }
    if (defaults.fps) {
      setFps(Number(defaults.fps));
    }
    if (defaults.quality) {
      setQuality(defaults.quality);
    }
    if (defaults.resolution) {
      setResolution(defaults.resolution);
    }
    if (Object.prototype.hasOwnProperty.call(defaults, "negativePrompt")) {
      setNegativePrompt(defaults.negativePrompt ?? "");
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
  const canSubmit = Boolean(activeProject && prompt.trim() && supportsMode && implementedMode && hasInputs && !presetMissingLoras.length);
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

  const localJobs = localJobIds
    .map((id) => jobs.find((job) => job.id === id))
    .filter((job) => job && job.status !== "completed");

  async function submit(event) {
    event.preventDefault();
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
    if (job?.id) {
      setLocalJobIds((ids) => [job.id, ...ids.filter((id) => id !== job.id)].slice(0, 4));
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
          {mode === "image_to_video" || mode === "first_last_frame" ? (
            <label>
              First frame
              <select onChange={(event) => setSourceAssetId(event.target.value)} value={sourceAssetId}>
                <option value="">Select image</option>
                {imageAssets.map((asset) => (
                  <option key={asset.id} value={asset.id}>
                    {asset.displayName}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {mode === "first_last_frame" ? (
            <label>
              Last frame
              <select onChange={(event) => setLastFrameAssetId(event.target.value)} value={lastFrameAssetId}>
                <option value="">Select image</option>
                {imageAssets.map((asset) => (
                  <option key={asset.id} value={asset.id}>
                    {asset.displayName}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {mode === "extend_clip" ? (
            <label>
              Source clip
              <select onChange={(event) => setSourceClipAssetId(event.target.value)} value={sourceClipAssetId}>
                <option value="">Select clip</option>
                {videoAssets.map((asset) => (
                  <option key={asset.id} value={asset.id}>
                    {asset.displayName}
                  </option>
                ))}
              </select>
            </label>
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
              Preset
              <select disabled={!availableRecipePresets.length} onChange={(event) => setRecipePresetId(event.target.value)} value={recipePresetId}>
                {availableRecipePresets.length ? (
                  availableRecipePresets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name ?? preset.id}
                    </option>
                  ))
                ) : (
                  <option value="">No presets</option>
                )}
              </select>
            </label>
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
              <strong>Presets unavailable</strong>
              <span>Generation can continue without preset defaults.</span>
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
          {presetMissingLoras.length ? (
            <p className="inline-warning">
              Preset cannot run until LoRA import finishes: {presetMissingLoras.map((lora) => lora.id).join(", ")}. Wait for the Queue or choose another preset.
            </p>
          ) : null}
          <button className="primary-action" disabled={!canSubmit} type="submit">
            {mode === "replace_person" ? "Replace Person" : "Generate Clip"}
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
          ) : localJobs.length ? null : (
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
