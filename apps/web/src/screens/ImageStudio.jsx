import React, { useEffect, useState } from "react";
import { AssetCard } from "../components/assetPanels.jsx";

export function ImageStudio({
  activeProject,
  assets,
  createImageJob,
  deleteAsset,
  purgeAsset,
  gpuOptions,
  imageModels,
  latestAssets,
  onPreview,
  requestedGpu,
  selectedAsset,
  setRequestedGpu,
  updateAssetStatus,
}) {
  const [mode, setMode] = useState("text_to_image");
  const [prompt, setPrompt] = useState("A cinematic frame of a neon street at midnight");
  const [stylePreset, setStylePreset] = useState("cinematic");
  const [count, setCount] = useState(4);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [model, setModel] = useState(imageModels[0]?.id ?? "z_image_turbo");
  const [seed, setSeed] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [resolution, setResolution] = useState("1024x1024");
  const [sourceAssetId, setSourceAssetId] = useState(selectedAsset?.id ?? "");

  useEffect(() => {
    if (!imageModels.some((item) => item.id === model)) {
      setModel(imageModels[0]?.id ?? "z_image_turbo");
    }
  }, [imageModels, model]);

  useEffect(() => {
    if (mode === "edit_image" && selectedAsset?.id) {
      setSourceAssetId(selectedAsset.id);
    }
  }, [mode, selectedAsset?.id]);

  const availableModels = imageModels.filter((item) => {
    const caps = item.capabilities ?? [];
    if (mode === "edit_image") {
      return caps.includes("edit_image") || caps.includes("image_edit");
    }
    return item.type === "image";
  });
  const [width, height] = resolution.split("x").map((value) => Number(value));

  function submit(event) {
    event.preventDefault();
    createImageJob({
      mode,
      prompt,
      negativePrompt,
      model,
      count,
      seed: seed === "" ? null : Number(seed),
      width,
      height,
      stylePreset,
      sourceAssetId: mode === "edit_image" ? sourceAssetId || null : null,
      loras: [],
      advanced: { resolution },
    });
  }

  return (
    <section className="main-surface image-studio">
      <div className="surface-header">
        <div className="section-heading">
          <p className="eyebrow">Image Studio</p>
          <h2>{activeProject ? activeProject.name : "Create a project"}</h2>
        </div>
        <div className="segmented-control" role="tablist" aria-label="Image mode">
          {[
            ["text_to_image", "Text"],
            ["edit_image", "Edit"],
            ["character_image", "Character"],
            ["style_variations", "Variations"],
          ].map(([value, label]) => (
            <button className={mode === value ? "active" : ""} key={value} onClick={() => setMode(value)} type="button">
              {label}
            </button>
          ))}
        </div>
      </div>

      <form className="studio-layout" onSubmit={submit}>
        <section className="studio-controls">
          {mode === "edit_image" ? (
            <label>
              Source
              <select onChange={(event) => setSourceAssetId(event.target.value)} value={sourceAssetId}>
                <option value="">Select image</option>
                {assets
                  .filter((asset) => asset.type === "image")
                  .map((asset) => (
                    <option key={asset.id} value={asset.id}>
                      {asset.displayName}
                    </option>
                  ))}
              </select>
            </label>
          ) : null}

          <label className="prompt-field">
            Prompt
            <textarea onChange={(event) => setPrompt(event.target.value)} value={prompt} />
          </label>

          <div className="control-grid">
            <label>
              Style
              <select onChange={(event) => setStylePreset(event.target.value)} value={stylePreset}>
                <option value="cinematic">Cinematic</option>
                <option value="photoreal">Photoreal</option>
                <option value="anime">Anime</option>
                <option value="fantasy">Fantasy</option>
                <option value="product">Product Shot</option>
              </select>
            </label>
            <label>
              Count
              <input min="1" max="8" onChange={(event) => setCount(Number(event.target.value))} type="number" value={count} />
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

          <button className="advanced-toggle" onClick={() => setAdvancedOpen((value) => !value)} type="button">
            {advancedOpen ? "Hide advanced" : "Advanced"}
          </button>

          {advancedOpen ? (
            <div className="advanced-panel">
              <label>
                Model
                <select onChange={(event) => setModel(event.target.value)} value={model}>
                  {(availableModels.length ? availableModels : imageModels).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Seed
                <input onChange={(event) => setSeed(event.target.value)} placeholder="Random" type="number" value={seed} />
              </label>
              <label>
                Resolution
                <select onChange={(event) => setResolution(event.target.value)} value={resolution}>
                  <option value="768x768">768 x 768</option>
                  <option value="1024x1024">1024 x 1024</option>
                  <option value="1280x720">1280 x 720</option>
                  <option value="720x1280">720 x 1280</option>
                </select>
              </label>
              <label className="prompt-field">
                Negative prompt
                <textarea onChange={(event) => setNegativePrompt(event.target.value)} value={negativePrompt} />
              </label>
            </div>
          ) : null}

          <button className="primary-action" disabled={!activeProject || !prompt.trim()} type="submit">
            Generate
          </button>
        </section>

        <section className="review-panel">
          <div className="section-heading">
            <p className="eyebrow">Fresh batch</p>
            <h2>Review</h2>
          </div>
          {latestAssets.length ? (
            <div className="review-grid">
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
            <div className="empty-panel">No fresh image batch</div>
          )}
        </section>
      </form>
    </section>
  );
}
