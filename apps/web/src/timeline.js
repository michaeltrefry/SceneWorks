export const aspectOptions = {
  "16:9": { width: 1280, height: 720, label: "16:9" },
  "9:16": { width: 720, height: 1280, label: "9:16" },
  "1:1": { width: 1024, height: 1024, label: "1:1" },
};
export const transitionOptions = ["cut", "crossfade", "fade_from_black", "fade_to_black"];
export const speedPresets = [0.25, 0.5, 1, 2];

export function createLocalTimeline(project, name = "Main timeline", aspectRatio = "16:9") {
  const dimensions = aspectOptions[aspectRatio] ?? aspectOptions["16:9"];
  return {
    schemaVersion: 1,
    id: `timeline_${crypto.randomUUID().replaceAll("-", "")}`,
    projectId: project.id,
    name,
    aspectRatio,
    width: dimensions.width,
    height: dimensions.height,
    fps: 30,
    duration: 0,
    tracks: [
      { id: "track_main", name: "Main", kind: "video", locked: false, muted: false, items: [] },
      { id: "track_overlay", name: "Overlay", kind: "overlay", locked: false, muted: false, items: [] },
      { id: "track_audio", name: "Audio", kind: "audio", locked: false, muted: false, items: [] },
    ],
    transitions: [],
    createdAt: null,
    updatedAt: null,
  };
}

export function timelineDuration(timeline) {
  return Math.max(0, ...timeline.tracks.flatMap((track) => track.items.map((item) => Number(item.timelineEnd) || 0)));
}

export function itemDuration(item) {
  return Math.max(0.1, Number(item.timelineEnd) - Number(item.timelineStart));
}

export function trackItems(track) {
  return [...track.items].sort((a, b) => a.timelineStart - b.timelineStart);
}

export function sourceTimestampAtPlayhead(item, playheadSeconds) {
  const start = Number(item.timelineStart) || 0;
  const end = Math.max(start, Number(item.timelineEnd) || start);
  const clamped = Math.min(Math.max(Number(playheadSeconds) || start, start), end);
  return (Number(item.sourceIn) || 0) + (clamped - start) * (Number(item.speed) || 1);
}

export function ensureItemVersionFields(item) {
  const versionAssetIds = Array.from(new Set([...(item.versionAssetIds ?? []), item.assetId].filter(Boolean)));
  return {
    ...item,
    currentVersionAssetId: item.currentVersionAssetId ?? item.assetId,
    versionAssetIds,
    versionHistory:
      item.versionHistory?.length > 0
        ? item.versionHistory
        : [{ assetId: item.assetId, createdAt: null, source: "original", jobId: null, note: null }],
  };
}
