import React, { useEffect, useState } from "react";
import { AssetCard } from "../components/assetPanels.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";

export function VideoStudio({
  activeProject,
  assets,
  createVideoJob,
  deleteAsset,
  purgeAsset,
  gpuOptions,
  latestAssets,
  onPreview,
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
  const selectedModel = videoModels.find((item) => item.id === model) ?? videoModels[0];
  const [duration, setDuration] = useState(selectedModel?.defaults?.duration ?? 6);
  const [resolution, setResolution] = useState(selectedModel?.defaults?.resolution ?? "768x512");
  const [fps, setFps] = useState(selectedModel?.defaults?.fps ?? 25);
  const [seed, setSeed] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.type === "image" ? selectedAsset.id : "");
  const [lastFrameAssetId, setLastFrameAssetId] = useState("");
  const [sourceClipAssetId, setSourceClipAssetId] = useState(selectedAsset?.type === "video" ? selectedAsset.id : "");

  useEffect(() => {
    if (!videoModels.some((item) => item.id === model)) {
      setModel(videoModels[0]?.id ?? "ltx_2_3");
    }
  }, [videoModels, model]);

  useEffect(() => {
    if (selectedAsset?.type === "image") {
      setSourceAssetId(selectedAsset.id);
    }
    if (selectedAsset?.type === "video") {
      setSourceClipAssetId(selectedAsset.id);
    }
  }, [selectedAsset?.id, selectedAsset?.type]);

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

  const modeOptions = [
    ["image_to_video", "Image to Video"],
    ["text_to_video", "Text to Video"],
    ["first_last_frame", "First/Last Frame"],
    ["extend_clip", "Extend Clip"],
    ["replace_person", "Replace Person"],
  ];
  const capabilities = selectedModel?.capabilities ?? [];
  const supportsMode = capabilities.includes(mode);
  const implementedMode = ["image_to_video", "text_to_video", "first_last_frame"].includes(mode);
  const hasInputs =
    mode === "text_to_video" ||
    (mode === "image_to_video" && sourceAssetId) ||
    (mode === "first_last_frame" && sourceAssetId && lastFrameAssetId) ||
    (mode === "extend_clip" && sourceClipAssetId) ||
    mode === "replace_person";
  const canSubmit = Boolean(activeProject && prompt.trim() && supportsMode && implementedMode && hasInputs);
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

  function submit(event) {
    event.preventDefault();
    createVideoJob({
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
      sourceAssetId: ["image_to_video", "first_last_frame"].includes(mode) ? sourceAssetId || null : null,
      lastFrameAssetId: mode === "first_last_frame" ? lastFrameAssetId || null : null,
      sourceClipAssetId: mode === "extend_clip" ? sourceClipAssetId || null : null,
      loras: [],
      advanced: { resolution, durationHint },
    });
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
            <div className="empty-panel compact-panel">Replacement is staged as a placeholder mode.</div>
          ) : null}

          <label className="prompt-field">
            Prompt
            <textarea onChange={(event) => setPrompt(event.target.value)} value={prompt} />
          </label>

          <div className="control-grid">
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
          <button className="primary-action" disabled={!canSubmit} type="submit">
            Generate Clip
          </button>
        </section>

        <section className="review-panel video-review">
          <div className="section-heading">
            <p className="eyebrow">Fresh clip</p>
            <h2>Review</h2>
          </div>
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
          ) : (
            <div className="empty-panel">No fresh video clip</div>
          )}

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
