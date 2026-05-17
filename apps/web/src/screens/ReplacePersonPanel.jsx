import React from "react";
import { AssetMedia } from "../components/assetMedia.jsx";

export function findReplacementModel(videoModels) {
  return videoModels.find((item) => item.capabilities?.includes("replace_person")) ?? null;
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

      <div className="guidance-strip">
        <strong>V1 placeholder tracking</strong>
        <span>Candidate boxes, tracking, and replacement output are procedural previews until a model adapter is connected.</span>
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
          <span>Box corrections are represented in the track sidecar; mask correction is deferred for the model adapter.</span>
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
