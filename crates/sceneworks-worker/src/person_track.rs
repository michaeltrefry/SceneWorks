//! Content-based selected-person tracking (sc-3634, Slice 2 of sc-3488 / epic 3482).
//!
//! Ports the Python `person_adapters.py` tracking semantics to Rust, replacing the procedural
//! `track_frames_from_detection` placeholder. The Python path runs `ultralytics yolo.track(
//! tracker="bytetrack.yaml")` — a pure algorithm (no neural net): a YOLO detector per frame plus
//! Kalman + IoU association. Here we run the native-MLX YOLO11 detector (sc-3633) at the 2-FPS
//! sample cadence and associate the per-frame boxes into track identities with a self-contained
//! SORT/ByteTrack-style tracker (constant-velocity prediction + two-stage greedy IoU matching; no
//! learned ReID — unnecessary for the few-people, forgiving-cadence person-track use case).
//!
//! Everything here is pure geometry/cadence/association logic, unit-tested without weights or
//! video. The orchestration (frame extraction → detect → these helpers → sidecar) lives in
//! `media_jobs::run_person_track`.

use serde_json::{json, Value};

/// Tracking sample cadence (frames per second of source). Matches the V1 sidecar cadence so
/// existing track consumers keep working. Mirrors Python `PERSON_TRACK_SAMPLE_RATE_FPS`.
pub(crate) const SAMPLE_RATE_FPS: f64 = 2.0;
pub(crate) const MIN_SAMPLES: usize = 3;
pub(crate) const MAX_SAMPLES: usize = 24;
/// A track frame whose confidence falls below this is flagged for correction (box still recorded).
const TRACK_LOW_CONFIDENCE: f64 = 0.40;

/// Match IoU floor for `choose_target_track_id` (Python default).
const TARGET_MIN_IOU: f64 = 0.1;

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn round_to(value: f64, places: i32) -> f64 {
    let factor = 10f64.powi(places);
    (value * factor).round() / factor
}

/// A normalized (0..1) `x/y/width/height` box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NormalizedBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl NormalizedBox {
    pub(crate) fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Build from a sidecar/JSON `{x,y,width,height}` object, clamping to 0..1.
    pub(crate) fn from_json(value: &Value) -> Self {
        let f = |key: &str| clamp01(value.get(key).and_then(Value::as_f64).unwrap_or(0.0));
        Self::new(f("x"), f("y"), f("width"), f("height"))
    }

    fn to_json(self) -> Value {
        json!({
            "x": round_to(self.x, 4),
            "y": round_to(self.y, 4),
            "width": round_to(self.width, 4),
            "height": round_to(self.height, 4),
        })
    }

    fn center(self) -> (f64, f64) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
}

/// Convert pixel `xyxy` to a normalized box (used to bring detector boxes into 0..1 space).
pub(crate) fn xyxy_to_normalized(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    width: u32,
    height: u32,
) -> NormalizedBox {
    if width == 0 || height == 0 {
        return NormalizedBox::new(0.0, 0.0, 0.0, 0.0);
    }
    let (w, h) = (width as f64, height as f64);
    let (left, right) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
    let (top, bottom) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
    NormalizedBox::new(
        clamp01(left / w),
        clamp01(top / h),
        clamp01((right - left) / w),
        clamp01((bottom - top) / h),
    )
}

/// Intersection-over-union of two normalized boxes.
pub(crate) fn box_iou(a: NormalizedBox, b: NormalizedBox) -> f64 {
    let (ax2, ay2) = (a.x + a.width, a.y + a.height);
    let (bx2, by2) = (b.x + b.width, b.y + b.height);
    let inter_w = (ax2.min(bx2) - a.x.max(b.x)).max(0.0);
    let inter_h = (ay2.min(by2) - a.y.max(b.y)).max(0.0);
    let intersection = inter_w * inter_h;
    let union = a.width * a.height + b.width * b.height - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Number of evenly spaced samples for a clip of `duration` seconds (Python `sample_count_for_duration`).
pub(crate) fn sample_count_for_duration(duration: f64) -> usize {
    let raw = (duration.max(0.0) * SAMPLE_RATE_FPS).round() as i64;
    (raw.max(MIN_SAMPLES as i64) as usize).min(MAX_SAMPLES)
}

/// Evenly spaced sample timestamps across the clip, inclusive of both ends (Python `sample_timestamps`).
pub(crate) fn sample_timestamps(duration: f64) -> Vec<f64> {
    let count = sample_count_for_duration(duration);
    let span = duration.max(0.0);
    if count <= 1 || span <= 0.0 {
        return vec![0.0];
    }
    (0..count)
        .map(|index| round_to(span * index as f64 / (count - 1) as f64, 4))
        .collect()
}

/// Person boxes observed in one sampled frame, keyed by tracker id (Python `FrameObservation`).
#[derive(Clone, Debug)]
pub(crate) struct FrameObservation {
    pub timestamp: f64,
    /// `(track_id, box, confidence)` for each person assigned an identity this frame.
    pub boxes: Vec<(i64, NormalizedBox, f64)>,
}

impl FrameObservation {
    fn get(&self, track_id: i64) -> Option<(NormalizedBox, f64)> {
        self.boxes
            .iter()
            .find(|(id, _, _)| *id == track_id)
            .map(|(_, b, c)| (*b, *c))
    }
}

// ---------------------------------------------------------------------------
// SORT/ByteTrack-style tracker: constant-velocity prediction + greedy IoU
// ---------------------------------------------------------------------------

/// One tracked identity: a constant-velocity estimate of the box center + size.
#[derive(Clone, Copy, Debug)]
struct Track {
    id: i64,
    cx: f64,
    cy: f64,
    w: f64,
    h: f64,
    vcx: f64,
    vcy: f64,
    time_since_update: u32,
}

impl Track {
    fn from_box(id: i64, b: NormalizedBox) -> Self {
        let (cx, cy) = b.center();
        Self {
            id,
            cx,
            cy,
            w: b.width,
            h: b.height,
            vcx: 0.0,
            vcy: 0.0,
            time_since_update: 0,
        }
    }

    /// Advance the constant-velocity estimate one frame and age the track.
    fn predict(&mut self) {
        self.cx += self.vcx;
        self.cy += self.vcy;
        self.time_since_update += 1;
    }

    /// Predicted box at the current (post-`predict`) estimate.
    fn predicted_box(&self) -> NormalizedBox {
        NormalizedBox::new(
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.w,
            self.h,
        )
    }

    /// Correct the estimate toward an observed box (velocity = blended center delta).
    fn update(&mut self, b: NormalizedBox) {
        let (cx, cy) = b.center();
        // Blend the velocity estimate (alpha=0.5) so a single jittery sample doesn't dominate.
        self.vcx = 0.5 * self.vcx + 0.5 * (cx - self.cx);
        self.vcy = 0.5 * self.vcy + 0.5 * (cy - self.cy);
        self.cx = cx;
        self.cy = cy;
        self.w = b.width;
        self.h = b.height;
        self.time_since_update = 0;
    }
}

/// A self-contained SORT/ByteTrack-style multi-object tracker over normalized boxes.
pub(crate) struct PersonTracker {
    tracks: Vec<Track>,
    next_id: i64,
    /// Detections at/above this confidence start new tracks + match first (ByteTrack high stage).
    high_thresh: f64,
    /// Detections below `min_conf` are ignored entirely.
    min_conf: f64,
    /// IoU floor for the high-confidence association round (loose — 2-FPS motion + prediction).
    match_thresh_high: f64,
    /// IoU floor for the low-confidence recovery round (stricter).
    match_thresh_low: f64,
    /// Frames a track survives without a detection before it is dropped.
    max_age: u32,
}

impl Default for PersonTracker {
    fn default() -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            high_thresh: 0.5,
            min_conf: 0.1,
            match_thresh_high: 0.2,
            match_thresh_low: 0.5,
            max_age: 2,
        }
    }
}

impl PersonTracker {
    /// Associate one frame's detections (`(box, confidence)`) into track ids. Returns the
    /// `(track_id, box, confidence)` for every detection assigned an identity this frame (the
    /// observed detection box, matching the Python tracker's recorded box).
    pub(crate) fn update(
        &mut self,
        detections: &[(NormalizedBox, f64)],
    ) -> Vec<(i64, NormalizedBox, f64)> {
        for track in &mut self.tracks {
            track.predict();
        }

        let high: Vec<usize> = (0..detections.len())
            .filter(|&i| detections[i].1 >= self.high_thresh)
            .collect();
        let low: Vec<usize> = (0..detections.len())
            .filter(|&i| detections[i].1 < self.high_thresh && detections[i].1 >= self.min_conf)
            .collect();

        let mut track_taken = vec![false; self.tracks.len()];
        let mut assignments: Vec<(i64, NormalizedBox, f64)> = Vec::new();

        // Round 1: high-confidence detections vs all tracks.
        let matched_high =
            self.greedy_match(&high, detections, &mut track_taken, self.match_thresh_high);
        for (det_idx, track_idx) in &matched_high {
            self.tracks[*track_idx].update(detections[*det_idx].0);
            assignments.push((
                self.tracks[*track_idx].id,
                detections[*det_idx].0,
                detections[*det_idx].1,
            ));
        }
        let matched_high_dets: Vec<usize> = matched_high.iter().map(|(d, _)| *d).collect();

        // Round 2: still-unmatched tracks vs low-confidence detections (recovery).
        let matched_low =
            self.greedy_match(&low, detections, &mut track_taken, self.match_thresh_low);
        for (det_idx, track_idx) in &matched_low {
            self.tracks[*track_idx].update(detections[*det_idx].0);
            assignments.push((
                self.tracks[*track_idx].id,
                detections[*det_idx].0,
                detections[*det_idx].1,
            ));
        }

        // Unmatched high-confidence detections start new tracks.
        for &det_idx in &high {
            if matched_high_dets.contains(&det_idx) {
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;
            self.tracks.push(Track::from_box(id, detections[det_idx].0));
            assignments.push((id, detections[det_idx].0, detections[det_idx].1));
        }

        // Drop tracks that have gone too long without a detection.
        let max_age = self.max_age;
        self.tracks
            .retain(|track| track.time_since_update <= max_age);
        assignments
    }

    /// Greedy IoU matching of `det_indices` to currently-free tracks, highest IoU first.
    fn greedy_match(
        &self,
        det_indices: &[usize],
        detections: &[(NormalizedBox, f64)],
        track_taken: &mut [bool],
        thresh: f64,
    ) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for &det_idx in det_indices {
            for (track_idx, track) in self.tracks.iter().enumerate() {
                if track_taken[track_idx] {
                    continue;
                }
                let iou = box_iou(detections[det_idx].0, track.predicted_box());
                if iou >= thresh {
                    pairs.push((iou, det_idx, track_idx));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut det_taken: Vec<usize> = Vec::new();
        let mut matched: Vec<(usize, usize)> = Vec::new();
        for (_iou, det_idx, track_idx) in pairs {
            if track_taken[track_idx] || det_taken.contains(&det_idx) {
                continue;
            }
            track_taken[track_idx] = true;
            det_taken.push(det_idx);
            matched.push((det_idx, track_idx));
        }
        matched
    }
}

/// Build the per-frame observations for a sequence of sampled frames' detections, running the
/// tracker over them in time order.
pub(crate) fn observe(frames: Vec<(f64, Vec<(NormalizedBox, f64)>)>) -> Vec<FrameObservation> {
    let mut tracker = PersonTracker::default();
    frames
        .into_iter()
        .map(|(timestamp, detections)| FrameObservation {
            timestamp,
            boxes: tracker.update(&detections),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Target selection + track assembly (Python `choose_target_track_id` / `assemble_track`)
// ---------------------------------------------------------------------------

fn nearest_observation(
    observations: &[FrameObservation],
    timestamp: f64,
) -> Option<&FrameObservation> {
    observations.iter().min_by(|a, b| {
        (a.timestamp - timestamp)
            .abs()
            .partial_cmp(&(b.timestamp - timestamp).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Match the user-selected detection to a tracker identity by IoU at the observation nearest the
/// selection frame. `None` when nothing overlaps (track honestly instead of locking onto the wrong
/// person).
pub(crate) fn choose_target_track_id(
    observations: &[FrameObservation],
    selected_box: NormalizedBox,
    selected_timestamp: f64,
) -> Option<i64> {
    let observation = nearest_observation(observations, selected_timestamp)?;
    let mut best_id: Option<i64> = None;
    let mut best_iou = TARGET_MIN_IOU;
    for (track_id, box_, _conf) in &observation.boxes {
        let score = box_iou(*box_, selected_box);
        if score >= best_iou {
            best_iou = score;
            best_id = Some(*track_id);
        }
    }
    best_id
}

/// One assembled track frame (Python `TrackFrame`).
#[derive(Clone, Debug)]
pub(crate) struct TrackFrame {
    pub timestamp: f64,
    pub box_: NormalizedBox,
    pub confidence: f64,
    pub detected: bool,
    pub flags: Vec<&'static str>,
}

impl TrackFrame {
    fn to_json(&self) -> Value {
        let mut payload = json!({
            "timestamp": round_to(self.timestamp, 4),
            "box": self.box_.to_json(),
            "confidence": round_to(self.confidence, 4),
            "detected": self.detected,
            "mask": Value::Null,
        });
        if !self.flags.is_empty() {
            payload["flags"] = json!(self.flags);
        }
        payload
    }
}

/// The resampled track plus its quality summary (Python `TrackAssembly`).
pub(crate) struct TrackAssembly {
    pub frames: Vec<TrackFrame>,
    pub target_track_id: Option<i64>,
    pub detected_frames: usize,
    pub quality: Value,
}

/// Resample the chosen tracker identity onto the requested sample cadence. Detected frames carry
/// the tracker's real box/confidence; frames where the target is absent are recorded
/// `detected=false` and flagged — never fabricated (Python `assemble_track`).
pub(crate) fn assemble_track(
    observations: &[FrameObservation],
    selected_box: NormalizedBox,
    selected_timestamp: f64,
    timestamps: &[f64],
) -> TrackAssembly {
    let target_id = choose_target_track_id(observations, selected_box, selected_timestamp);
    let mut frames: Vec<TrackFrame> = Vec::with_capacity(timestamps.len());
    let mut detected_count = 0usize;
    let mut lost_frames: Vec<usize> = Vec::new();
    let mut last_box = selected_box;

    for (index, &stamp) in timestamps.iter().enumerate() {
        let entry = target_id
            .and_then(|id| nearest_observation(observations, stamp).and_then(|obs| obs.get(id)));
        match entry {
            Some((box_, confidence)) => {
                last_box = box_;
                let flags = if confidence < TRACK_LOW_CONFIDENCE {
                    vec!["low_confidence"]
                } else {
                    Vec::new()
                };
                frames.push(TrackFrame {
                    timestamp: stamp,
                    box_,
                    confidence,
                    detected: true,
                    flags,
                });
                detected_count += 1;
            }
            None => {
                lost_frames.push(index);
                frames.push(TrackFrame {
                    timestamp: stamp,
                    box_: last_box,
                    confidence: 0.0,
                    detected: false,
                    flags: vec!["lost_target"],
                });
            }
        }
    }

    let sampled = timestamps.len();
    let quality = json!({
        "trackId": target_id,
        "sampledFrames": sampled,
        "detectedFrames": detected_count,
        "lostFrames": lost_frames,
        "detectedRatio": if sampled > 0 { round_to(detected_count as f64 / sampled as f64, 4) } else { 0.0 },
    });
    TrackAssembly {
        frames,
        target_track_id: target_id,
        detected_frames: detected_count,
        quality,
    }
}

/// Serialize assembled frames to the sidecar `frames` array shape (Python `TrackFrame.to_dict`).
pub(crate) fn frames_to_json(frames: &[TrackFrame]) -> Vec<Value> {
    frames.iter().map(TrackFrame::to_json).collect()
}

/// Average confidence over detected frames (for `status.averageConfidence`).
pub(crate) fn average_confidence(frames: &[TrackFrame]) -> f64 {
    let detected: Vec<f64> = frames
        .iter()
        .filter(|f| f.detected)
        .map(|f| f.confidence)
        .collect();
    if detected.is_empty() {
        0.0
    } else {
        round_to(detected.iter().sum::<f64>() / detected.len() as f64, 4)
    }
}

#[cfg(test)]
mod tests;
