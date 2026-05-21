import React from "react";
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

function maskStateCopy(track) {
  const state = track?.status?.maskState;
  return MASK_STATE_COPY[state] ?? "Tracked boxes are stored in the track sidecar.";
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
  videoAssets,
}) {
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

      <div className="replace-actions">
        <button disabled={!sourceClipAssetId} onClick={analyzeSource} type="button">
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
        <button disabled={!representativeFrame || !selectedDetection} onClick={createTrack} type="button">
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
