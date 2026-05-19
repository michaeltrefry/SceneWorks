import React, { useEffect, useMemo, useState } from "react";
import { AssetMedia, assetCanRenderAsImage } from "../components/assetMedia.jsx";
import { formatSeconds } from "../formatting.js";
import {
  aspectOptions,
  ensureItemVersionFields,
  itemDuration,
  sourceTimestampAtPlayhead,
  speedPresets,
  timelineDuration,
  trackItems,
  transitionOptions,
} from "../timeline.js";

export function EditorScreen({
  activeProject,
  activeTimeline,
  assets,
  createTimeline,
  extractTimelineFrame,
  exportTimeline,
  onPreview,
  onSendImage,
  onSendVideo,
  queueTimelineVideoJob,
  saveTimeline,
  selectedTimelineId,
  setActiveTimeline,
  setSelectedTimelineId,
  timelines,
}) {
  const [newTimelineName, setNewTimelineName] = useState("Main timeline");
  const [newAspectRatio, setNewAspectRatio] = useState("16:9");
  const [selectedItemId, setSelectedItemId] = useState(null);
  const [addDuration, setAddDuration] = useState(4);
  const [exportResolution, setExportResolution] = useState(720);
  const [history, setHistory] = useState([]);
  const [future, setFuture] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playheadSeconds, setPlayheadSeconds] = useState(0);
  const [generationPrompt, setGenerationPrompt] = useState("Continue the action with matching motion and lighting");
  const [extensionDuration, setExtensionDuration] = useState(4);
  const [timelineNotice, setTimelineNotice] = useState("");

  const selectedItem = useMemo(() => {
    if (!activeTimeline) {
      return null;
    }
    return activeTimeline.tracks.flatMap((track) => track.items).find((item) => item.id === selectedItemId) ?? null;
  }, [activeTimeline, selectedItemId]);
  const selectedAsset = useMemo(() => assets.find((asset) => asset.id === selectedItem?.assetId) ?? null, [assets, selectedItem]);
  const duration = activeTimeline ? timelineDuration(activeTimeline) : 0;
  const timelineScale = Math.max(12, duration + 4);
  const mainAssets = assets.filter((asset) => asset.type === "video" || asset.file?.mimeType?.startsWith("video/"));
  const stillAssets = assets.filter((asset) => assetCanRenderAsImage(asset));

  useEffect(() => {
    setHistory([]);
    setFuture([]);
    setSelectedItemId(null);
  }, [activeTimeline?.id]);

  useEffect(() => {
    if (selectedItem) {
      setPlayheadSeconds(Number(selectedItem.timelineStart) || 0);
    }
    setTimelineNotice("");
  }, [selectedItem?.id]);

  useEffect(() => {
    function onKeyDown(event) {
      const target = event.target;
      const isTyping = ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName);
      if (isTyping) {
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        setIsPlaying((value) => !value);
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        if (selectedItemId) {
          event.preventDefault();
          removeSelectedItem();
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeTimeline, history, future, selectedItemId]);

  async function submitNewTimeline(event) {
    event.preventDefault();
    await createTimeline({ name: newTimelineName, aspectRatio: newAspectRatio, fps: 30 });
  }

  function commit(nextTimeline) {
    if (!activeTimeline) {
      return;
    }
    setHistory((items) => [...items.slice(-24), activeTimeline]);
    setFuture([]);
    setActiveTimeline({ ...nextTimeline, duration: timelineDuration(nextTimeline) });
  }

  function updateTimelineItem(itemId, changes) {
    if (!activeTimeline) {
      return;
    }
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((track) => ({
        ...track,
        items: track.items.map((item) => (item.id === itemId ? normalizeTimelineItem({ ...item, ...changes }) : item)),
      })),
    });
  }

  function undo() {
    if (!history.length || !activeTimeline) {
      return;
    }
    const previous = history[history.length - 1];
    setHistory((items) => items.slice(0, -1));
    setFuture((items) => [activeTimeline, ...items]);
    setActiveTimeline(previous);
  }

  function redo() {
    if (!future.length || !activeTimeline) {
      return;
    }
    const next = future[0];
    setFuture((items) => items.slice(1));
    setHistory((items) => [...items, activeTimeline]);
    setActiveTimeline(next);
  }

  function addAssetToTrack(asset, trackId = "track_main") {
    if (!activeTimeline) {
      return;
    }
    const isStill = asset.type !== "video" && assetCanRenderAsImage(asset);
    const track = activeTimeline.tracks.find((item) => item.id === trackId) ?? activeTimeline.tracks[0];
    const start = Math.max(0, ...track.items.map((item) => item.timelineEnd));
    const sourceDuration = Number(asset.file?.duration) || Number(addDuration) || 4;
    const durationSeconds = isStill ? Number(addDuration) || 4 : sourceDuration;
    const item = normalizeTimelineItem({
      id: `item_${crypto.randomUUID().replaceAll("-", "")}`,
      trackId: track.id,
      assetId: asset.id,
      type: isStill ? "image" : "video",
      displayName: asset.displayName,
      sourceIn: 0,
      sourceOut: Math.max(0.1, sourceDuration),
      timelineStart: start,
      timelineEnd: start + Math.max(0.1, durationSeconds),
      speed: 1,
      fit: "fit",
      volume: 1,
      versionAssetIds: [asset.id],
      currentVersionAssetId: asset.id,
      versionHistory: [{ assetId: asset.id, createdAt: null, source: "original", jobId: null, note: null }],
      transitionIn: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
      transitionOut: { id: `transition_${crypto.randomUUID().replaceAll("-", "")}`, type: "cut", duration: 0 },
    });
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((current) =>
        current.id === track.id ? { ...current, items: [...current.items, item] } : current,
      ),
    });
    setSelectedItemId(item.id);
  }

  function removeSelectedItem() {
    if (!activeTimeline || !selectedItemId) {
      return;
    }
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((track) => ({
        ...track,
        items: track.items.filter((item) => item.id !== selectedItemId),
      })),
    });
    setSelectedItemId(null);
  }

  function changeItemTrack(trackId) {
    if (!activeTimeline || !selectedItem) {
      return;
    }
    commit({
      ...activeTimeline,
      tracks: activeTimeline.tracks.map((track) => {
        if (track.id === selectedItem.trackId) {
          return { ...track, items: track.items.filter((item) => item.id !== selectedItem.id) };
        }
        if (track.id === trackId) {
          return { ...track, items: [...track.items, normalizeTimelineItem({ ...selectedItem, trackId })] };
        }
        return track;
      }),
    });
  }

  function normalizeTimelineItem(item) {
    const start = Number(item.timelineStart) || 0;
    const end = Math.max(start + 0.1, Number(item.timelineEnd) || start + 0.1);
    const sourceIn = Number(item.sourceIn) || 0;
    const sourceOut = Math.max(sourceIn + 0.1, Number(item.sourceOut) || sourceIn + itemDuration(item));
    return {
      ...ensureItemVersionFields(item),
      sourceIn,
      sourceOut,
      timelineStart: Math.max(0, start),
      timelineEnd: end,
      speed: Math.max(0.1, Number(item.speed) || 1),
    };
  }

  function selectedTrackItems() {
    if (!activeTimeline || !selectedItem) {
      return [];
    }
    const track = activeTimeline.tracks.find((item) => item.id === selectedItem.trackId);
    return track ? trackItems(track) : [];
  }

  function nextItemAfterSelection() {
    return selectedTrackItems().find((item) => item.timelineStart >= selectedItem.timelineEnd && item.id !== selectedItem.id) ?? null;
  }

  function generationBaseContext(action, extra = {}) {
    return {
      action,
      timelineId: activeTimeline.id,
      timelineName: activeTimeline.name,
      itemId: selectedItem.id,
      trackId: selectedItem.trackId,
      sourceAssetId: selectedItem.assetId,
      sourceTimestamp: sourceTimestampAtPlayhead(selectedItem, playheadSeconds),
      ...extra,
    };
  }

  async function extractFrame(intendedUse = "reuse") {
    if (!selectedItem || selectedItem.type !== "video") {
      setTimelineNotice("Select a video clip before extracting a frame.");
      return;
    }
    const job = await extractTimelineFrame({ timeline: activeTimeline, item: selectedItem, playheadSeconds, intendedUse });
    if (job) {
      setTimelineNotice("Frame extraction queued.");
    }
  }

  async function extendSelectedClip() {
    if (!selectedItem || selectedItem.type !== "video") {
      setTimelineNotice("Select a video clip before extending.");
      return;
    }
    const duration = Math.min(15, Math.max(1, Number(extensionDuration) || itemDuration(selectedItem)));
    const job = await queueTimelineVideoJob({
      mode: "extend_clip",
      prompt: generationPrompt,
      model: "ltx_2_3",
      duration,
      fps: activeTimeline.fps,
      width: activeTimeline.width,
      height: activeTimeline.height,
      quality: "balanced",
      sourceClipAssetId: selectedItem.assetId,
      loras: [],
      advanced: {
        resolution: `${activeTimeline.width}x${activeTimeline.height}`,
        timelineAction: "extend",
        timelineContext: generationBaseContext("extend", {
          endpointTimestamp: Number(selectedItem.sourceOut) || sourceTimestampAtPlayhead(selectedItem, selectedItem.timelineEnd),
          timelineStart: Number(selectedItem.timelineEnd),
        }),
      },
    });
    if (job) {
      setTimelineNotice("Extension job queued. The new clip will land after the selection when it completes.");
    }
  }

  async function replaceSelectedItem() {
    if (!selectedItem) {
      return;
    }
    const isStill = selectedItem.type === "image";
    const duration = itemDuration(selectedItem);
    const job = await queueTimelineVideoJob({
      mode: isStill ? "image_to_video" : "extend_clip",
      prompt: generationPrompt,
      model: "ltx_2_3",
      duration,
      fps: activeTimeline.fps,
      width: activeTimeline.width,
      height: activeTimeline.height,
      quality: "balanced",
      sourceAssetId: isStill ? selectedItem.assetId : null,
      sourceClipAssetId: isStill ? null : selectedItem.assetId,
      loras: [],
      advanced: {
        resolution: `${activeTimeline.width}x${activeTimeline.height}`,
        timelineAction: "replace",
        timelineContext: generationBaseContext("replace"),
      },
    });
    if (job) {
      setTimelineNotice("Replacement job queued. The prior asset will stay in this item's version history.");
    }
  }

  async function bridgeAfterSelectedClip() {
    if (!selectedItem || selectedItem.type !== "video") {
      setTimelineNotice("Select the left video clip before generating a bridge.");
      return;
    }
    const rightItem = nextItemAfterSelection();
    if (!rightItem || rightItem.type !== "video") {
      setTimelineNotice("Place another video clip after this one on the same track first.");
      return;
    }
    const gap = Number(rightItem.timelineStart) - Number(selectedItem.timelineEnd);
    if (gap <= 0.05) {
      setTimelineNotice("Create space between the clips before generating a bridge.");
      return;
    }
    const job = await queueTimelineVideoJob({
      mode: "video_bridge",
      prompt: generationPrompt,
      model: "ltx_2_3",
      duration: Number(gap.toFixed(2)),
      fps: activeTimeline.fps,
      width: activeTimeline.width,
      height: activeTimeline.height,
      quality: "balanced",
      sourceClipAssetId: selectedItem.assetId,
      bridgeRightClipAssetId: rightItem.assetId,
      loras: [],
      advanced: {
        resolution: `${activeTimeline.width}x${activeTimeline.height}`,
        timelineAction: "bridge",
        timelineContext: generationBaseContext("bridge", {
          rightItemId: rightItem.id,
          rightAssetId: rightItem.assetId,
          leftTimestamp: Number(selectedItem.sourceOut) || sourceTimestampAtPlayhead(selectedItem, selectedItem.timelineEnd),
          rightTimestamp: Number(rightItem.sourceIn) || 0,
          timelineStart: Number(selectedItem.timelineEnd),
          timelineEnd: Number(rightItem.timelineStart),
        }),
      },
    });
    if (job) {
      setTimelineNotice("Bridge job queued. The generated clip will land in the gap.");
    }
  }

  if (!activeProject) {
    return (
      <section className="main-surface">
        <div className="section-heading">
          <p className="eyebrow">Editor</p>
          <h2>Create a project</h2>
        </div>
        <div className="empty-panel">Open a project before assembling a timeline.</div>
      </section>
    );
  }

  return (
    <section className="main-surface editor-surface">
      <div className="surface-header editor-header hero">
        <div className="section-heading">
          <p className="eyebrow">Editor</p>
          <h2>{activeTimeline?.name ?? "Timelines"}</h2>
          <p className="hero-blurb">
            Sequence clips on the timeline, scrub through the preview, and export when the cut feels right.
          </p>
        </div>
        <div className="editor-actions">
          <select onChange={(event) => setSelectedTimelineId(event.target.value)} value={selectedTimelineId ?? ""}>
            <option value="">Select timeline</option>
            {timelines.map((timeline) => (
              <option key={timeline.id} value={timeline.id}>
                {timeline.name}
              </option>
            ))}
          </select>
          <button disabled={!activeTimeline} onClick={() => saveTimeline(activeTimeline)} type="button">
            Save
          </button>
          <button disabled={!history.length} onClick={undo} type="button">
            Undo
          </button>
          <button disabled={!future.length} onClick={redo} type="button">
            Redo
          </button>
        </div>
      </div>

      <form className="timeline-create" onSubmit={submitNewTimeline}>
        <label>
          Timeline
          <input onChange={(event) => setNewTimelineName(event.target.value)} value={newTimelineName} />
        </label>
        <label>
          Aspect
          <select onChange={(event) => setNewAspectRatio(event.target.value)} value={newAspectRatio}>
            {Object.entries(aspectOptions).map(([value, option]) => (
              <option key={value} value={value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <button type="submit">New Timeline</button>
      </form>

      {activeTimeline ? (
        <div className="editor-layout">
          <section className="editor-preview">
            <div className={`preview-canvas aspect-${activeTimeline.aspectRatio.replace(":", "-")}`}>
              {selectedAsset ? <AssetMedia asset={selectedAsset} /> : <span>Select a timeline item</span>}
            </div>
            <div className="playback-bar">
              <button onClick={() => setIsPlaying((value) => !value)} type="button">
                {isPlaying ? "Pause" : "Play"}
              </button>
              <span>{formatSeconds(Math.round(duration))}</span>
              <span>{activeTimeline.aspectRatio}</span>
              <span>{activeTimeline.fps} fps</span>
            </div>
          </section>

          <aside className="editor-inspector">
            {selectedItem ? (
              <>
                <div className="section-heading">
                  <p className="eyebrow">Clip</p>
                  <h2>{selectedItem.displayName}</h2>
                </div>
                <label>
                  Track
                  <select onChange={(event) => changeItemTrack(event.target.value)} value={selectedItem.trackId}>
                    {activeTimeline.tracks.map((track) => (
                      <option key={track.id} value={track.id}>
                        {track.name}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="control-grid compact-controls">
                  <label>
                    Start
                    <input
                      min="0"
                      onChange={(event) => updateTimelineItem(selectedItem.id, { timelineStart: Number(event.target.value) })}
                      step="0.1"
                      type="number"
                      value={selectedItem.timelineStart}
                    />
                  </label>
                  <label>
                    End
                    <input
                      min="0.1"
                      onChange={(event) => updateTimelineItem(selectedItem.id, { timelineEnd: Number(event.target.value) })}
                      step="0.1"
                      type="number"
                      value={selectedItem.timelineEnd}
                    />
                  </label>
                  <label>
                    Source In
                    <input
                      min="0"
                      onChange={(event) => updateTimelineItem(selectedItem.id, { sourceIn: Number(event.target.value) })}
                      step="0.1"
                      type="number"
                      value={selectedItem.sourceIn}
                    />
                  </label>
                  <label>
                    Source Out
                    <input
                      min="0.1"
                      onChange={(event) => updateTimelineItem(selectedItem.id, { sourceOut: Number(event.target.value) })}
                      step="0.1"
                      type="number"
                      value={selectedItem.sourceOut}
                    />
                  </label>
                </div>
                <label>
                  Speed
                  <select onChange={(event) => updateTimelineItem(selectedItem.id, { speed: Number(event.target.value) })} value={selectedItem.speed}>
                    {speedPresets.map((speed) => (
                      <option key={speed} value={speed}>
                        {speed}x
                      </option>
                    ))}
                    {!speedPresets.includes(Number(selectedItem.speed)) ? <option value={selectedItem.speed}>Custom {selectedItem.speed}x</option> : null}
                  </select>
                </label>
                <label>
                  Custom speed
                  <input
                    min="0.1"
                    onChange={(event) => updateTimelineItem(selectedItem.id, { speed: Number(event.target.value) })}
                    step="0.05"
                    type="number"
                    value={selectedItem.speed}
                  />
                </label>
                <div className="generation-hook-panel">
                  <label>
                    Playhead
                    <input
                      max={selectedItem.timelineEnd}
                      min={selectedItem.timelineStart}
                      onChange={(event) => setPlayheadSeconds(Number(event.target.value))}
                      step="0.1"
                      type="number"
                      value={playheadSeconds}
                    />
                  </label>
                  <label className="prompt-field">
                    Generation prompt
                    <textarea onChange={(event) => setGenerationPrompt(event.target.value)} value={generationPrompt} />
                  </label>
                  <label>
                    Extension duration
                    <input
                      max="15"
                      min="1"
                      onChange={(event) => setExtensionDuration(Number(event.target.value))}
                      step="0.5"
                      type="number"
                      value={extensionDuration}
                    />
                  </label>
                  <div className="hook-actions">
                    <button disabled={selectedItem.type !== "video"} onClick={() => extractFrame("reuse")} type="button">
                      Extract Frame
                    </button>
                    <button disabled={selectedItem.type === "video"} onClick={() => onSendImage(selectedAsset)} type="button">
                      Send Image
                    </button>
                    <button onClick={() => onSendVideo(selectedAsset)} type="button">
                      Send Video
                    </button>
                    <button disabled={selectedItem.type !== "video"} onClick={extendSelectedClip} type="button">
                      Extend
                    </button>
                    <button onClick={replaceSelectedItem} type="button">
                      Replace
                    </button>
                    <button disabled={selectedItem.type !== "video"} onClick={bridgeAfterSelectedClip} type="button">
                      Bridge Gap
                    </button>
                  </div>
                  <div className="version-list">
                    <strong>{ensureItemVersionFields(selectedItem).versionAssetIds.length} versions</strong>
                    {ensureItemVersionFields(selectedItem)
                      .versionAssetIds.slice(-4)
                      .map((assetId) => (
                        <button
                          className={assetId === selectedItem.assetId ? "active" : ""}
                          key={assetId}
                          onClick={() => updateTimelineItem(selectedItem.id, { assetId, currentVersionAssetId: assetId })}
                          type="button"
                        >
                          {assetId === selectedItem.assetId ? "Current" : "Restore"} {assetId.slice(-6)}
                        </button>
                      ))}
                  </div>
                  {timelineNotice ? <p className="inline-warning">{timelineNotice}</p> : null}
                </div>
                <label>
                  Transition in
                  <select
                    onChange={(event) =>
                      updateTimelineItem(selectedItem.id, {
                        transitionIn: { ...(selectedItem.transitionIn ?? {}), type: event.target.value },
                      })
                    }
                    value={selectedItem.transitionIn?.type ?? "cut"}
                  >
                    {transitionOptions.map((transition) => (
                      <option key={transition} value={transition}>
                        {transition.replaceAll("_", " ")}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Transition out
                  <select
                    onChange={(event) =>
                      updateTimelineItem(selectedItem.id, {
                        transitionOut: { ...(selectedItem.transitionOut ?? {}), type: event.target.value },
                      })
                    }
                    value={selectedItem.transitionOut?.type ?? "cut"}
                  >
                    {transitionOptions.map((transition) => (
                      <option key={transition} value={transition}>
                        {transition.replaceAll("_", " ")}
                      </option>
                    ))}
                  </select>
                </label>
                <button className="danger-action" onClick={removeSelectedItem} type="button">
                  Delete Clip
                </button>
              </>
            ) : (
              <div className="empty-panel compact-panel">No clip selected</div>
            )}
          </aside>

          <section className="timeline-panel">
            <div className="timeline-ruler">
              <span>0s</span>
              <span>{formatSeconds(Math.ceil(timelineScale / 2))}</span>
              <span>{formatSeconds(Math.ceil(timelineScale))}</span>
            </div>
            <div className="timeline-tracks">
              {activeTimeline.tracks.map((track) => (
                <div className="timeline-track" key={track.id}>
                  <strong>{track.name}</strong>
                  <div className="track-lane">
                    {trackItems(track).map((item) => (
                      <button
                        className={selectedItemId === item.id ? "timeline-item active" : "timeline-item"}
                        key={item.id}
                        onClick={() => setSelectedItemId(item.id)}
                        style={{
                          left: `${(item.timelineStart / timelineScale) * 100}%`,
                          width: `${Math.max(4, (itemDuration(item) / timelineScale) * 100)}%`,
                        }}
                        type="button"
                      >
                        <span>{item.displayName}</span>
                        <small>{item.speed}x</small>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <aside className="asset-bin">
            <div className="bin-controls">
              <label>
                Still duration
                <input min="0.5" onChange={(event) => setAddDuration(Number(event.target.value))} step="0.5" type="number" value={addDuration} />
              </label>
            </div>
            <div className="asset-bin-list">
              {[...mainAssets, ...stillAssets].slice(0, 18).map((asset) => (
                <article className="bin-asset" key={asset.id}>
                  <button onClick={() => onPreview(asset)} type="button">
                    <AssetMedia asset={asset} />
                  </button>
                  <strong>{asset.displayName}</strong>
                  <div className="bin-actions">
                    <button onClick={() => addAssetToTrack(asset, "track_main")} type="button">
                      Main
                    </button>
                    <button onClick={() => addAssetToTrack(asset, "track_overlay")} type="button">
                      Overlay
                    </button>
                  </div>
                </article>
              ))}
              {assets.length === 0 ? <div className="empty-panel compact-panel">No media assets</div> : null}
            </div>
          </aside>

          <form
            className="export-strip"
            onSubmit={(event) => {
              event.preventDefault();
              exportTimeline(activeTimeline, { resolution: Number(exportResolution), fps: activeTimeline.fps });
            }}
          >
            <label>
              MP4 height
              <select onChange={(event) => setExportResolution(Number(event.target.value))} value={exportResolution}>
                {[640, 720, 1024, 1280].map((resolution) => (
                  <option key={resolution} value={resolution}>
                    {resolution}
                  </option>
                ))}
              </select>
            </label>
            <button className="primary-action" disabled={!activeTimeline.tracks.some((track) => track.items.length)} type="submit">
              Export MP4
            </button>
          </form>
        </div>
      ) : (
        <div className="empty-panel">Create a timeline to start editing.</div>
      )}
    </section>
  );
}
