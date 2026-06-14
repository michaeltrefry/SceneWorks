import React, { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL } from "../api.js";
import { AssetPickerField } from "../components/AssetPicker.jsx";
import { AssetMedia } from "../components/assetMedia.jsx";

export function findReplacementModel(videoModels) {
  return videoModels.find((item) => item.capabilities?.includes("replace_person")) ?? null;
}

const MASK_STATE_COPY = {
  active: "Per-frame segmentation masks generated for tracked frames.",
  generated: "Per-frame segmentation masks generated for most tracked frames.",
  degraded: "Box-derived masks (segmentation backend unavailable on the worker).",
  missing: "No masks yet — re-run tracking or correct the track.",
  deferred: "Procedural preview track — not real tracking output.",
};

const BOX_FIELDS = [
  ["x", "X"],
  ["y", "Y"],
  ["width", "W"],
  ["height", "H"],
];

function maskStateCopy(track) {
  const state = track?.status?.maskState;
  return MASK_STATE_COPY[state] ?? "Tracked boxes are stored in the track sidecar.";
}

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function roundComponent(value) {
  return Math.round(clamp01(value) * 10000) / 10000;
}

function normalizeBox(box) {
  return {
    x: roundComponent(box?.x ?? 0),
    y: roundComponent(box?.y ?? 0),
    width: roundComponent(box?.width ?? 0),
    height: roundComponent(box?.height ?? 0),
  };
}

function boxesEqual(a, b) {
  if (!a || !b) {
    return false;
  }
  return BOX_FIELDS.every(([key]) => Math.abs((a[key] ?? 0) - (b[key] ?? 0)) < 1e-4);
}

function maskUrl(projectId, relPath) {
  if (!projectId || !relPath) {
    return "";
  }
  const normalized = String(relPath).replaceAll("\\", "/");
  return `${API_BASE_URL}/api/v1/projects/${projectId}/files/${normalized}`;
}

/**
 * Sparse correction surface for a tracked person (sc-1485). Scrub sampled
 * tracking frames, inspect the box/mask overlay, nudge a box, and reject low
 * quality frames; the corrections persist in the track sidecar and the
 * replacement pipeline regenerates masks from corrected boxes.
 */
function PersonTrackCorrections({ track, sourceClip, saveTrackCorrections }) {
  const frames = useMemo(() => track?.frames ?? [], [track]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [drafts, setDrafts] = useState({});
  const [saving, setSaving] = useState(false);
  const videoRef = useRef(null);

  const correctionsSignature = JSON.stringify(track?.corrections ?? []);
  // Seed working drafts from the persisted corrections so reopening a track
  // shows its saved adjustments, and re-seed after a save converges the UI.
  useEffect(() => {
    const seeded = {};
    for (const correction of track?.corrections ?? []) {
      const index = correction?.frameIndex;
      if (!Number.isInteger(index) || index < 0 || index >= frames.length) {
        continue;
      }
      seeded[index] = {
        box: correction.box ? normalizeBox(correction.box) : null,
        rejected: Boolean(correction.rejected),
      };
    }
    setDrafts(seeded);
    setFrameIndex((current) => (current < frames.length ? current : 0));
  }, [track?.id, correctionsSignature, frames.length]);

  useEffect(() => {
    const video = videoRef.current;
    const timestamp = frames[frameIndex]?.timestamp;
    if (!video || !Number.isFinite(timestamp)) {
      return;
    }
    try {
      video.currentTime = timestamp;
    } catch {
      // Seeking before metadata loads is retried by onLoadedMetadata below.
    }
  }, [frameIndex, frames]);

  if (!frames.length) {
    return (
      <div className="empty-panel compact-panel">
        This track has no sampled frames to correct yet.
      </div>
    );
  }

  const safeIndex = Math.min(frameIndex, frames.length - 1);
  const frame = frames[safeIndex];
  const draft = drafts[safeIndex];
  const workingBox = normalizeBox(draft?.box ?? frame.box ?? {});
  const rejected = Boolean(draft?.rejected);
  const flags = frame.flags ?? [];
  const overlayMask = frame.mask ? maskUrl(track.projectId, frame.mask) : "";

  function updateDraft(index, updater) {
    setDrafts((current) => {
      const base = current[index] ?? {
        box: normalizeBox(frames[index]?.box ?? {}),
        rejected: false,
      };
      const next = updater({ box: base.box ? normalizeBox(base.box) : normalizeBox(frames[index]?.box ?? {}), rejected: base.rejected });
      return { ...current, [index]: next };
    });
  }

  function setBoxComponent(key, rawValue) {
    const value = roundComponent(Number.parseFloat(rawValue));
    updateDraft(safeIndex, (entry) => ({ ...entry, box: { ...entry.box, [key]: value } }));
  }

  function setRejected(value) {
    updateDraft(safeIndex, (entry) => ({ ...entry, rejected: value }));
  }

  function resetFrame() {
    setDrafts((current) => {
      const next = { ...current };
      delete next[safeIndex];
      return next;
    });
  }

  // The corrections payload is the UI's full view: only frames whose box drifted
  // from the tracked box or that are rejected carry intent worth persisting.
  const pendingCorrections = Object.entries(drafts)
    .map(([key, entry]) => {
      const index = Number(key);
      const original = frames[index]?.box;
      const boxChanged = entry.box && original && !boxesEqual(entry.box, original);
      const isRejected = Boolean(entry.rejected);
      if (!boxChanged && !isRejected) {
        return null;
      }
      const correction = { frameIndex: index, rejected: isRejected, author: "ui", source: "manual" };
      if (boxChanged) {
        correction.box = normalizeBox(entry.box);
      }
      return correction;
    })
    .filter(Boolean)
    .sort((a, b) => a.frameIndex - b.frameIndex);

  const persistedCount = (track?.corrections ?? []).length;
  const frameCorrected = pendingCorrections.some((correction) => correction.frameIndex === safeIndex);

  // Compare the working set against what is persisted (ignoring stamped
  // author/createdAt/source) so Save only lights up when something changed.
  const comparable = (correction) =>
    JSON.stringify({
      frameIndex: correction.frameIndex,
      box: correction.box ? normalizeBox(correction.box) : null,
      rejected: Boolean(correction.rejected),
    });
  const persistedComparable = (track?.corrections ?? [])
    .filter((correction) => Number.isInteger(correction?.frameIndex))
    .map(comparable)
    .sort();
  const pendingComparable = pendingCorrections.map(comparable).sort();
  const dirty = JSON.stringify(persistedComparable) !== JSON.stringify(pendingComparable);

  async function save() {
    if (saving || typeof saveTrackCorrections !== "function") {
      return;
    }
    setSaving(true);
    try {
      await saveTrackCorrections(track.id, pendingCorrections);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="person-track-corrections" aria-label="Track corrections">
      <div className="person-correction-header">
        <strong>Review &amp; correct track</strong>
        <span>
          Frame {safeIndex + 1} / {frames.length}
          {Number.isFinite(frame.timestamp) ? ` • ${frame.timestamp.toFixed(2)}s` : ""}
          {Number.isFinite(frame.confidence) ? ` • ${Math.round(frame.confidence * 100)}% conf` : ""}
        </span>
      </div>

      <div className="person-selection-frame">
        {sourceClip ? (
          <AssetMedia
            asset={sourceClip}
            controls={false}
            muted
            onLoadedMetadata={() => {
              const video = videoRef.current;
              if (video && Number.isFinite(frame.timestamp)) {
                try {
                  video.currentTime = frame.timestamp;
                } catch {
                  // Ignore: some browsers reject seeks until the buffer is ready.
                }
              }
            }}
            ref={videoRef}
          />
        ) : null}
        {overlayMask ? <img alt="" className="person-track-mask-overlay" src={overlayMask} /> : null}
        <div
          className={rejected ? "person-box rejected" : "person-box active"}
          style={{
            left: `${workingBox.x * 100}%`,
            top: `${workingBox.y * 100}%`,
            width: `${workingBox.width * 100}%`,
            height: `${workingBox.height * 100}%`,
          }}
        >
          <span>{rejected ? "rejected" : "corrected box"}</span>
        </div>
      </div>

      <div className="person-correction-scrubber">
        <button
          aria-label="Previous frame"
          disabled={safeIndex <= 0}
          onClick={() => setFrameIndex(Math.max(0, safeIndex - 1))}
          type="button"
        >
          ‹
        </button>
        <input
          aria-label="Scrub tracking frames"
          max={frames.length - 1}
          min={0}
          onChange={(event) => setFrameIndex(Number.parseInt(event.target.value, 10) || 0)}
          step={1}
          type="range"
          value={safeIndex}
        />
        <button
          aria-label="Next frame"
          disabled={safeIndex >= frames.length - 1}
          onClick={() => setFrameIndex(Math.min(frames.length - 1, safeIndex + 1))}
          type="button"
        >
          ›
        </button>
      </div>

      {flags.length ? (
        <div className="person-correction-flags" role="status">
          {flags.map((flag) => (
            <span className="person-correction-flag" key={flag}>
              {flag.replaceAll("_", " ")}
            </span>
          ))}
        </div>
      ) : null}

      <div className="control-grid person-correction-box">
        {BOX_FIELDS.map(([key, label]) => (
          <label key={key}>
            {label}
            <input
              aria-label={`Box ${key}`}
              disabled={rejected}
              max={1}
              min={0}
              onChange={(event) => setBoxComponent(key, event.target.value)}
              step={0.01}
              type="number"
              value={workingBox[key]}
            />
          </label>
        ))}
      </div>

      <label className="person-correction-reject">
        <input checked={rejected} onChange={(event) => setRejected(event.target.checked)} type="checkbox" />
        Reject this frame (low quality — replacement borrows the nearest good frame)
      </label>

      <div className="guidance-strip">
        <span>
          Adjusting a box regenerates that frame&apos;s mask from the corrected box at replacement time. Saved
          corrections record author and time in the track sidecar.
        </span>
      </div>

      <div className="replace-actions">
        <div className="person-correction-actions">
          <button disabled={!frameCorrected} onClick={resetFrame} type="button">
            Reset frame
          </button>
          <button className="primary" disabled={saving || !dirty} onClick={save} type="button">
            {saving ? "Saving…" : "Save corrections"}
          </button>
        </div>
        <span>
          {dirty ? `${pendingCorrections.length} unsaved` : `${persistedCount} saved`}
        </span>
      </div>
    </div>
  );
}

export function ReplacePersonPanel({
  createPersonDetectionJob,
  createPersonTrackJob,
  detectionResult,
  matchingTracks,
  representativeFrame,
  selectedDetection,
  selectedTrack,
  setPersonTrackId,
  setReplacementMode,
  setSelectedDetectionId,
  setSourceClipAssetId,
  setTrackName,
  sourceClipAssetId,
  trackName,
  personTrackId,
  replacementMode,
  saveTrackCorrections,
  videoAssets,
  videoModels = [],
  model,
  setModel,
  personReadiness = {},
}) {
  // The replacement backend = the replace-capable video models. The user picks one here (it drives
  // the job's `model`); SCAIL-2 (scail2_14b) is the native cross-identity engine, the others inpaint
  // the masked region via Wan-VACE (sc-5449 / sc-5452).
  const replacementModels = useMemo(
    () => videoModels.filter((item) => item.capabilities?.includes("replace_person")),
    [videoModels],
  );
  // Default-open: only gate when readiness explicitly reports a backend missing.
  const detectReady = personReadiness?.detect?.ready !== false;
  const trackReady = personReadiness?.track?.ready !== false;
  const replaceReady = personReadiness?.replace?.ready !== false;
  const readinessNotice = !detectReady
    ? "Detection unavailable: start a GPU worker with the detector backend installed (apps/worker/requirements-person.txt)."
    : !trackReady
      ? "Tracking unavailable: no live GPU worker is advertising the tracker capability."
      : !replaceReady
        ? "Replacement unavailable: no live GPU worker can run person replacement yet."
        : "";

  const selectedTrackSourceClip = selectedTrack
    ? videoAssets.find((asset) => asset.id === selectedTrack.sourceAssetId) ?? null
    : null;

  function analyzeSource() {
    if (!sourceClipAssetId) {
      return;
    }
    createPersonDetectionJob({ sourceAssetId: sourceClipAssetId }, { navigateToQueue: false });
  }

  function createTrack() {
    if (!sourceClipAssetId || !representativeFrame || !selectedDetection) {
      return;
    }
    createPersonTrackJob(
      {
        sourceAssetId: sourceClipAssetId,
        representativeFrameAssetId: representativeFrame.id,
        detection: selectedDetection,
        trackName,
      },
      { navigateToQueue: false },
    );
  }

  return (
    <div className="replace-person-panel">
      <AssetPickerField
        assets={videoAssets}
        buttonLabel="Select clip"
        emptyLabel="No source clip selected"
        label="Source clip"
        onChange={setSourceClipAssetId}
        value={sourceClipAssetId}
      />

      <div className="guidance-strip">
        <strong>Real person tracking</strong>
        <span>Detection and tracking run on a GPU worker (YOLO + ByteTrack). Replacement uses per-frame segmentation masks when a segmenter is installed and falls back to box masks otherwise.</span>
      </div>

      {readinessNotice ? (
        <div className="guidance-strip warning" role="status">
          <strong>Not ready</strong>
          <span>{readinessNotice}</span>
        </div>
      ) : null}

      <div className="replace-actions">
        <button disabled={!sourceClipAssetId || !detectReady} onClick={analyzeSource} type="button">
          Analyze Source
        </button>
        <span>{detectionResult ? `${detectionResult.detections?.length ?? 0} candidates` : "No analysis yet"}</span>
      </div>

      {representativeFrame ? (
        <div className="person-selection-frame">
          <AssetMedia asset={representativeFrame} />
          {(detectionResult?.detections ?? []).map((detection) => (
            <button
              aria-label={`Select ${detection.label}`}
              className={selectedDetection?.id === detection.id ? "person-box active" : "person-box"}
              key={detection.id}
              onClick={() => setSelectedDetectionId(detection.id)}
              style={{
                left: `${detection.box.x * 100}%`,
                top: `${detection.box.y * 100}%`,
                width: `${detection.box.width * 100}%`,
                height: `${detection.box.height * 100}%`,
              }}
              type="button"
            >
              <span>{Math.round(detection.confidence * 100)}%</span>
            </button>
          ))}
        </div>
      ) : (
        <div className="empty-panel compact-panel">Analyze a source clip to extract a selection frame.</div>
      )}

      <div className="control-grid compact-controls">
        <label>
          Track name
          <input onChange={(event) => setTrackName(event.target.value)} value={trackName} />
        </label>
        <button disabled={!representativeFrame || !selectedDetection || !trackReady} onClick={createTrack} type="button">
          Save Track
        </button>
      </div>

      <label>
        Person track
        <select onChange={(event) => setPersonTrackId(event.target.value)} value={personTrackId}>
          <option value="">Select tracked person</option>
          {matchingTracks.map((track) => (
            <option key={track.id} value={track.id}>
              {track.name}
            </option>
          ))}
        </select>
      </label>

      {selectedTrack ? (
        <div className="guidance-strip">
          <strong>{selectedTrack.status?.averageConfidence ? `${Math.round(selectedTrack.status.averageConfidence * 100)}% track` : "Reusable track"}</strong>
          <span>{maskStateCopy(selectedTrack)}</span>
        </div>
      ) : null}

      {selectedTrack ? (
        <PersonTrackCorrections
          key={selectedTrack.id}
          saveTrackCorrections={saveTrackCorrections}
          sourceClip={selectedTrackSourceClip}
          track={selectedTrack}
        />
      ) : null}

      {replacementModels.length > 1 && setModel ? (
        <label>
          Replacement engine
          <select onChange={(event) => setModel(event.target.value)} value={model}>
            {replacementModels.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      {model === "scail2_14b" ? (
        <div className="guidance-strip">
          <strong>SCAIL-2 full-character replacement</strong>
          <span>
            SCAIL-2 re-renders the whole tracked person from the character reference, so the
            Replacement mode below (face-only / keep-outfit) does not apply.
          </span>
        </div>
      ) : null}

      <label>
        Replacement mode
        <select onChange={(event) => setReplacementMode(event.target.value)} value={replacementMode}>
          <option value="face_only">Face Only</option>
          <option value="full_person_keep_outfit">Full Person, Keep Outfit</option>
          <option value="full_person_replace_outfit">Full Person, Replace Outfit</option>
        </select>
      </label>
    </div>
  );
}
