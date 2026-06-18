import React, { useMemo, useState } from "react";
import { useAppContext } from "../../context/AppContext.js";
import { Icon } from "../../components/Icons.jsx";
import { AssetMedia, AssetThumbnail, assetUrl } from "../../components/assetMedia.jsx";

const FILTERS = [
  { id: "all", label: "All" },
  { id: "image", label: "Pictures" },
  { id: "video", label: "Videos" },
];

function createdAtMs(asset) {
  const value = Date.parse(asset?.createdAt ?? "");
  return Number.isFinite(value) ? value : 0;
}

function dimsOf(asset, fallback = "1024x1024") {
  const width = asset?.file?.width;
  const height = asset?.file?.height;
  return width && height ? { width, height, resolution: `${width}x${height}` } : { ...{ width: 1024, height: 1024 }, resolution: fallback };
}

function titleOf(asset) {
  const prompt = asset?.recipe?.prompt ?? asset?.displayName ?? "";
  const trimmed = String(prompt).trim();
  if (!trimmed) return asset?.type === "video" ? "Untitled clip" : "Untitled picture";
  return trimmed.length > 60 ? `${trimmed.slice(0, 57)}…` : trimmed;
}

export function MyCreations() {
  const {
    recentImageAssets = [],
    recentVideoAssets = [],
    imageModels = [],
    videoModels = [],
    createImageJob,
    createVideoJob,
    updateAssetStatus,
    setPreviewAsset,
    setSelectedAssetId,
    setActiveView,
    setUiMode,
  } = useAppContext();

  const [filter, setFilter] = useState("all");
  const [selectedId, setSelectedId] = useState(null);
  const [notice, setNotice] = useState("");

  const items = useMemo(() => {
    const merged = [...recentImageAssets, ...recentVideoAssets];
    const filtered = filter === "all" ? merged : merged.filter((asset) => asset.type === filter);
    return filtered.slice().sort((a, b) => createdAtMs(b) - createdAtMs(a));
  }, [recentImageAssets, recentVideoAssets, filter]);

  const selected = useMemo(
    () => items.find((asset) => asset.id === selectedId) ?? items[0] ?? null,
    [items, selectedId],
  );

  const isImage = selected?.type === "image";
  const canRemake = isImage && Boolean(selected?.recipe?.prompt);
  const imageToVideoModel = useMemo(
    () => videoModels.find((model) => (model.capabilities ?? []).includes("image_to_video")) ?? null,
    [videoModels],
  );
  const canMakeVideoFromThis = isImage && Boolean(selected?.recipe?.prompt) && Boolean(imageToVideoModel);

  async function makeMoreLikeThis() {
    if (!canRemake) return;
    const recipe = selected.recipe;
    const dims = dimsOf(selected);
    setNotice("");
    const job = await createImageJob({
      mode: "text_to_image",
      prompt: recipe.prompt,
      negativePrompt: recipe.normalizedSettings?.negativePrompt ?? "",
      model: recipe.model ?? imageModels[0]?.id ?? "z_image_turbo",
      count: 4,
      width: dims.width,
      height: dims.height,
      recipePresetId: null,
      loras: [],
      advanced: { resolution: dims.resolution },
    });
    setNotice(job ? "Started — your new options will appear in Make a picture." : "Couldn't start that — try again.");
  }

  async function makeVideoFromThis() {
    if (!canMakeVideoFromThis) return;
    const model = imageToVideoModel;
    const dims = dimsOf(selected, "1280x720");
    setNotice("");
    const job = await createVideoJob({
      mode: "image_to_video",
      prompt: selected.recipe?.prompt ?? "",
      negativePrompt: "",
      model: model?.id ?? "",
      duration: Number(model?.defaults?.duration ?? 6),
      fps: Number(model?.defaults?.fps ?? 25),
      width: dims.width,
      height: dims.height,
      quality: "balanced",
      seed: null,
      recipePresetId: null,
      characterId: null,
      characterLookId: null,
      sourceAssetId: selected.id,
      fitMode: "crop",
      lastFrameAssetId: null,
      loras: [],
      advanced: { resolution: dims.resolution, motion: "slow push-in" },
    });
    setNotice(job ? "Started — your clip will appear in In progress." : "Couldn't start that — try again.");
  }

  function toggleSave() {
    updateAssetStatus?.(selected, { favorite: !selected?.status?.favorite });
  }

  function openInAdvanced() {
    setUiMode?.("advanced");
    setSelectedAssetId?.(selected.id);
    setActiveView?.("Library");
  }

  return (
    <section className="main-surface sw-make">
      <div className="sw-creations-grid">
        <div>
          <div className="sw-chips sw-filters">
            {FILTERS.map((entry) => (
              <button
                type="button"
                key={entry.id}
                className={`sw-chip ${filter === entry.id ? "on" : ""}`.trim()}
                aria-pressed={filter === entry.id}
                onClick={() => setFilter(entry.id)}
              >
                {entry.label}
              </button>
            ))}
          </div>

          {items.length === 0 ? (
            <div className="sw-empty">Nothing here yet — head to Make a picture or Make a video to start.</div>
          ) : (
            <div className="sw-gallery">
              {items.map((asset) => (
                <button
                  type="button"
                  key={asset.id}
                  className={`sw-gitem ${selected?.id === asset.id ? "sel" : ""}`.trim()}
                  onClick={() => setSelectedId(asset.id)}
                >
                  <AssetThumbnail asset={asset} />
                </button>
              ))}
            </div>
          )}
        </div>

        {selected ? (
          <aside className="sw-detail">
            <button type="button" className="sw-detail-preview" onClick={() => setPreviewAsset?.(selected)}>
              <AssetMedia asset={selected} controls={false} />
            </button>
            <h3 className="sw-detail-title">{titleOf(selected)}</h3>
            <p className="sw-detail-sub">
              {selected.type === "video" ? "Clip" : "Picture"}
              {selected.recipe?.model ? ` · ${selected.recipe.model}` : ""}
            </p>

            <div className="sw-detail-actions">
              {canRemake ? (
                <button type="button" className="sw-act primary" onClick={makeMoreLikeThis}>
                  <Icon.Image /> Make more like this
                </button>
              ) : null}
              {canMakeVideoFromThis ? (
                <button type="button" className="sw-act" onClick={makeVideoFromThis}>
                  <Icon.Video /> Make a video from this
                </button>
              ) : null}
              <button type="button" className="sw-act" onClick={toggleSave}>
                <Icon.Star filled={Boolean(selected.status?.favorite)} />
                {selected.status?.favorite ? "Saved" : "Save"}
              </button>
              <a className="sw-act" href={assetUrl(selected)} download>
                <Icon.ArrowRight /> Download
              </a>
            </div>

            {notice ? <p className="sw-notice">{notice}</p> : null}
            <button type="button" className="sw-advlink" onClick={openInAdvanced}>
              Need precise controls? Open in Advanced →
            </button>
          </aside>
        ) : null}
      </div>
    </section>
  );
}
