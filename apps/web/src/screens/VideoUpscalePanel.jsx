// Standalone video-upscale control for the Video Studio (epic 4811 / sc-4816) —
// SceneWorks' first video upscaler. Picks a source video clip and submits a
// `video_upscale` job (native-MLX SeedVR2 one-step super-resolution) via the generic
// /api/v1/jobs endpoint (createVideoUpscaleJob). It is a post-process on an existing
// clip, NOT a generation mode, so it lives in its own card rather than the mode picker.
import React, { useMemo, useState } from "react";

import { AssetPickerField } from "../components/AssetPicker.jsx";
import { DEFAULT_MAC_CAPABILITIES, macFeatureBlock } from "../macGating.js";

// SeedVR2 is the only Mac engine (no torch path). 7B (sc-5197) / int8 (sc-5198) are
// tracked engine follow-ups, so the variant is fixed to 3B here.
const VIDEO_UPSCALE_ENGINES = [{ id: "seedvr2", label: "SeedVR2", model: "seedvr2_3b", factors: [2, 4] }];

export function VideoUpscalePanel({
  videoAssets = [],
  selectedAsset = null,
  createVideoUpscaleJob,
  macCapabilities = DEFAULT_MAC_CAPABILITIES,
  onSubmitted,
}) {
  const engine = VIDEO_UPSCALE_ENGINES[0];
  const [sourceAssetId, setSourceAssetId] = useState(
    selectedAsset?.type === "video" ? selectedAsset.id : "",
  );
  const [factor, setFactor] = useState(2);
  const [softness, setSoftness] = useState(0);
  const [submitting, setSubmitting] = useState(false);

  const block = macFeatureBlock(macCapabilities, "videoUpscale");
  const source = useMemo(
    () => videoAssets.find((asset) => asset.id === sourceAssetId) ?? null,
    [videoAssets, sourceAssetId],
  );
  const srcW = source?.file?.width;
  const srcH = source?.file?.height;
  const targetLabel = srcW && srcH ? `${srcW * factor} × ${srcH * factor}` : null;
  const canSubmit = Boolean(sourceAssetId) && !block && !submitting;

  const onUpscale = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      const job = await createVideoUpscaleJob?.({
        sourceAssetId,
        factor,
        engine: engine.id,
        model: engine.model,
        softness,
        displayName: source?.displayName
          ? `${source.displayName} (${factor}x upscaled)`
          : undefined,
      });
      if (job && onSubmitted) onSubmitted(job);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <aside className="upscale-card" data-testid="video-upscale-panel">
      <h3>Upscale video</h3>
      <p className="upscale-card-sub">{engine.label} · one-step super-resolution (Mac / MLX)</p>
      {block ? <p className="mac-feature-block">{block.text}</p> : null}
      <AssetPickerField
        assets={videoAssets}
        buttonLabel="Select clip"
        emptyLabel="No source clip selected"
        label="Source clip"
        onChange={setSourceAssetId}
        value={sourceAssetId}
      />
      <div className="upscale-field">
        <span className="upscale-field-label">Scale</span>
        <div className="upscale-chip-row">
          {engine.factors.map((value) => (
            <button
              className={factor === value ? "motion-chip active" : "motion-chip"}
              disabled={Boolean(block)}
              key={value}
              onClick={() => setFactor(value)}
              type="button"
            >
              {value}×
            </button>
          ))}
        </div>
      </div>
      <label className="upscale-field">
        <span className="upscale-field-label">Softness {softness.toFixed(2)}</span>
        <input
          disabled={Boolean(block)}
          max="1"
          min="0"
          onChange={(event) => setSoftness(Number(event.target.value))}
          step="0.05"
          type="range"
          value={softness}
        />
      </label>
      {targetLabel ? <p className="upscale-card-sub">Output ≈ {targetLabel}</p> : null}
      <button className="prompt-cta" disabled={!canSubmit} onClick={onUpscale} type="button">
        {submitting ? "Upscaling…" : "Upscale clip"}
      </button>
    </aside>
  );
}
