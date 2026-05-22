"""Real person detection, tracking, and segmentation for the Replace Person workflow.

Epic sc-1090 originally shipped procedural placeholders in the Rust CPU utility
worker (``candidate_people`` / ``track_frames_from_detection``): hard-coded
candidate boxes and synthetic x-drift that never inspected video pixels. This
module replaces those with model-backed detection (sc-1480), content-based
tracking (sc-1481), and per-frame segmentation masks (sc-1482) running in the
Python GPU worker, where torch/ultralytics/SAM2 live.

Design notes
------------
* The heavy backends (ultralytics, SAM2/MatAnyone, torch, numpy) are imported
  lazily inside the model seams so this module stays importable on a host that
  only has Pillow + the standard library. That keeps the orchestration and the
  pure geometry/cadence helpers unit-testable without a GPU image, exactly like
  the video/training adapters.
* Geometry is normalized (0..1) box math in plain Python; only the model seams
  and mask rasterization touch numpy/PIL. Tests inject fake detectors/trackers/
  segmenters that read pixel content, so coverage proves the pipeline is
  content-dependent rather than template-driven.
* The orchestration functions return a result dict and write the same sidecar
  shapes the Rust worker produced, so the API/UI keep working — but successful
  real jobs set ``personDetectionActive``/``personTrackingActive`` to ``True``
  and record adapter/model/runtime metadata instead of the old inactive flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from PIL import Image, ImageDraw

from sceneworks_shared import (
    find_asset_sidecar_path,
    find_project_path,
    index_asset,
    read_json,
    safe_float,
    utc_now,
)

ProgressCallback = Callable[[str, str, float, str], None]
CancelCallback = Callable[[], bool]

# COCO "person" class index, shared by the Ultralytics detection and tracking models.
PERSON_CLASS_INDEX = 0
# Default confidence floor for accepting a detection/track box.
DEFAULT_DETECTION_CONFIDENCE = 0.25
# Representative-frame analysis resolution, matching the Rust placeholder frame size.
DETECTION_FRAME_WIDTH = 1280
DETECTION_FRAME_HEIGHT = 720
# Tracking sample cadence (frames per second of source). Matches the V1 sidecar
# cadence so existing track consumers keep working; replacement resamples as needed.
PERSON_TRACK_SAMPLE_RATE_FPS = 2.0
PERSON_TRACK_MIN_SAMPLES = 3
PERSON_TRACK_MAX_SAMPLES = 24
# A track frame whose confidence falls below this is flagged for correction
# rather than silently trusted; the box is still recorded honestly.
TRACK_LOW_CONFIDENCE = 0.40

DETECTOR_ADAPTER_ID = "ultralytics_person_detect"
TRACKER_ADAPTER_ID = "ultralytics_person_track"
SEGMENTER_ADAPTER_ID = "sam2_person_segment"


# ---------------------------------------------------------------------------
# Backend availability (advertised as worker capabilities only when installed)
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detector_backend_available() -> bool:
    """True when a real person detector (Ultralytics YOLO) can be imported."""
    return _module_available("ultralytics")


def tracker_backend_available() -> bool:
    """True when content-based tracking is available.

    Ultralytics ships ByteTrack/BoT-SORT trackers in-package, so the detector
    backend is also the tracker backend.
    """
    return _module_available("ultralytics")


def segmenter_backend_available() -> bool:
    """True when a video segmentation/matting backend is importable.

    SAM2 is the primary target; MatAnyone (Wan2GP's reference video matter) is an
    accepted alternative. Either presence advertises the segmenter capability;
    its absence degrades masks to boxes instead of failing the track job.
    """
    return _module_available("sam2") or _module_available("matanyone")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedBox:
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalizedBox":
        return cls(
            x=clamp01(safe_float(value.get("x"), 0.0, 0.0, 1.0)),
            y=clamp01(safe_float(value.get("y"), 0.0, 0.0, 1.0)),
            width=clamp01(safe_float(value.get("width"), 0.0, 0.0, 1.0)),
            height=clamp01(safe_float(value.get("height"), 0.0, 0.0, 1.0)),
        )


@dataclass(frozen=True)
class PersonDetection:
    id: str
    label: str
    box: NormalizedBox
    confidence: float

    def to_dict(self, *, frame_width: int, frame_height: int) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "box": self.box.to_dict(),
            "confidence": round(self.confidence, 4),
            "frameWidth": frame_width,
            "frameHeight": frame_height,
            "maskState": "missing",
        }


@dataclass
class TrackFrame:
    timestamp: float
    box: NormalizedBox
    confidence: float
    detected: bool = True
    mask: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": round(self.timestamp, 4),
            "box": self.box.to_dict(),
            "confidence": round(self.confidence, 4),
            "detected": self.detected,
            "mask": self.mask,
        }
        if self.flags:
            payload["flags"] = list(self.flags)
        return payload


# ---------------------------------------------------------------------------
# Pure geometry / cadence helpers (unit-tested directly, no models)
# ---------------------------------------------------------------------------


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def box_iou(a: NormalizedBox, b: NormalizedBox) -> float:
    """Intersection-over-union of two normalized boxes."""
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    inter_x1, inter_y1 = max(a.x, b.x), max(a.y, b.y)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def xyxy_to_normalized(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> NormalizedBox:
    """Convert pixel xyxy to a normalized x/y/width/height box."""
    if width <= 0 or height <= 0:
        return NormalizedBox(0.0, 0.0, 0.0, 0.0)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return NormalizedBox(
        x=clamp01(left / width),
        y=clamp01(top / height),
        width=clamp01((right - left) / width),
        height=clamp01((bottom - top) / height),
    )


def sample_count_for_duration(duration: float) -> int:
    raw = round(max(0.0, duration) * PERSON_TRACK_SAMPLE_RATE_FPS)
    return int(max(PERSON_TRACK_MIN_SAMPLES, min(PERSON_TRACK_MAX_SAMPLES, raw)))


def sample_timestamps(duration: float) -> list[float]:
    """Evenly spaced sample timestamps across the clip, inclusive of both ends."""
    count = sample_count_for_duration(duration)
    span = max(0.0, duration)
    if count <= 1 or span <= 0:
        return [0.0]
    return [round(span * index / (count - 1), 4) for index in range(count)]


def select_target_index(detections: list[PersonDetection], selected: NormalizedBox) -> int | None:
    """Pick the detection that best matches the user-selected box by IoU.

    Returns None when nothing overlaps, so callers can fail honestly instead of
    snapping to an unrelated person.
    """
    best_index: int | None = None
    best_iou = 0.0
    for index, detection in enumerate(detections):
        score = box_iou(detection.box, selected)
        if score > best_iou:
            best_iou = score
            best_index = index
    return best_index if best_iou > 0.0 else None


def frame_sha256(image: Image.Image) -> str:
    """Stable content hash of a frame, recorded in lineage so detection results
    can be tied to the exact pixels analyzed."""
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:".encode())
    digest.update(image.convert("RGB").tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Model seams (lazy heavy imports; monkeypatched in tests)
# ---------------------------------------------------------------------------


class DetectorRuntime(Protocol):
    """A loaded person detector. Real impl wraps an Ultralytics YOLO model."""

    model_ref: str
    runtime: dict[str, Any]

    def detect(self, image: Image.Image, *, confidence: float) -> list[PersonDetection]:
        ...


def resolve_detector_model(settings: Any, override: str | None) -> str:
    """Resolve the detector weights reference.

    Order: explicit override -> env -> local data-dir weight -> bundled default
    name (Ultralytics resolves/downloads bare names into its own cache).
    """
    if override:
        return override
    env_ref = os.getenv("SCENEWORKS_PERSON_DETECTOR_MODEL")
    if env_ref:
        return env_ref
    data_dir = getattr(settings, "data_dir", None)
    if data_dir is not None:
        for candidate in ("yolo11x.pt", "yolo11l.pt", "yolo11m.pt", "yolov8x.pt"):
            local = Path(data_dir) / "models" / "person-detect" / candidate
            if local.exists():
                return str(local)
    return "yolo11m.pt"


def load_person_detector(settings: Any, *, model_ref: str | None = None) -> DetectorRuntime:
    """Load a real Ultralytics person detector. Raises a clear error when the
    backend or weights are unavailable so the job fails honestly."""
    if not detector_backend_available():
        raise RuntimeError(
            "Person detection requires the Ultralytics backend. Install "
            "apps/worker/requirements-person.txt in the GPU worker image."
        )
    ultralytics = importlib.import_module("ultralytics")
    resolved = resolve_detector_model(settings, model_ref)
    yolo = ultralytics.YOLO(resolved)
    runtime = {
        "backend": "ultralytics",
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "device": getattr(settings, "gpu_id", "cpu"),
    }
    return _UltralyticsDetector(model=yolo, model_ref=resolved, runtime=runtime)


@dataclass
class _UltralyticsDetector:
    model: Any
    model_ref: str
    runtime: dict[str, Any]

    def detect(self, image: Image.Image, *, confidence: float) -> list[PersonDetection]:
        results = self.model.predict(
            source=image,
            classes=[PERSON_CLASS_INDEX],
            conf=confidence,
            verbose=False,
        )
        detections: list[PersonDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            for index, (coords, conf) in enumerate(zip(xyxy, confs)):
                box = xyxy_to_normalized(coords[0], coords[1], coords[2], coords[3], image.width, image.height)
                if box.width <= 0 or box.height <= 0:
                    continue
                detections.append(
                    PersonDetection(
                        id=f"person_{index + 1}",
                        label=f"Person {index + 1}",
                        box=box,
                        confidence=float(conf),
                    )
                )
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections


# ---------------------------------------------------------------------------
# Detection orchestration (sc-1480)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionRequest:
    project_id: str
    source_asset_id: str
    source_timestamp: float | None
    detector_model: str | None
    confidence: float


def _require(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Person detection job payload is missing '{key}'.")
    return value


def detection_request_from_job(job: dict[str, Any]) -> DetectionRequest:
    payload = job.get("payload") or {}
    advanced = payload.get("advanced") or {}
    timestamp = payload.get("sourceTimestamp")
    return DetectionRequest(
        project_id=_require(payload, "projectId"),
        source_asset_id=_require(payload, "sourceAssetId"),
        source_timestamp=None if timestamp is None else safe_float(timestamp, 0.0, 0.0, 3600.0),
        detector_model=advanced.get("detectorModel") or payload.get("detectorModel"),
        confidence=safe_float(advanced.get("confidence"), DEFAULT_DETECTION_CONFIDENCE, 0.01, 1.0),
    )


def _source_duration(project_path: Path, asset_id: str) -> float:
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is None:
        raise RuntimeError(f"Source clip asset not found: {asset_id}.")
    payload = read_json(sidecar_path)
    return safe_float(payload.get("file", {}).get("duration"), 6.0, 0.0, 3600.0)


def _source_display_name(project_path: Path, asset_id: str) -> str:
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is None:
        return "clip"
    payload = read_json(sidecar_path)
    return str(payload.get("displayName") or "clip")


def run_person_detect(
    settings: Any,
    job: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCallback | None = None,
    detector_factory: Callable[..., DetectorRuntime] = load_person_detector,
) -> dict[str, Any]:
    """Detect selectable people in a representative source frame using a real
    detector, persisting the selection-frame asset and returning pixel-derived
    candidates with adapter/model/runtime metadata (sc-1480)."""
    from .video_adapters import load_source_frame  # lazy: keeps this module light

    def report(stage: str, value: float, message: str) -> None:
        if progress is not None:
            progress("running", stage, value, message)

    def check_cancel() -> None:
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Person detection canceled.")

    request = detection_request_from_job(job)
    project_path = find_project_path(settings.data_dir / "recent-projects.json", request.project_id)
    duration = _source_duration(project_path, request.source_asset_id)
    timestamp = request.source_timestamp
    if timestamp is None:
        timestamp = round(duration * 0.25, 4) if duration > 0 else 0.0
    timestamp = float(max(0.0, min(timestamp, max(duration, 0.0))))

    report("extracting", 0.25, "Extracting representative frame.")
    check_cancel()
    frame = load_source_frame(
        project_path,
        request.source_asset_id,
        timestamp,
        DETECTION_FRAME_WIDTH,
        DETECTION_FRAME_HEIGHT,
    )
    if frame is None:
        raise RuntimeError(
            f"Could not extract a representative frame from source asset {request.source_asset_id}."
        )

    report("loading_model", 0.45, "Loading person detector.")
    check_cancel()
    detector = detector_factory(settings, model_ref=request.detector_model)

    report("running", 0.6, "Detecting people in representative frame.")
    detections = detector.detect(frame, confidence=request.confidence)

    report("saving", 0.82, "Saving representative frame and candidate boxes.")
    check_cancel()
    asset_id = f"asset_{uuid4().hex}"
    created_at = utc_now()
    media_rel = f"assets/frames/{created_at[:10]}_person-frame_{asset_id[-8:]}.png"
    media_path = project_path / media_rel
    media_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(media_path, format="PNG")
    source_hash = frame_sha256(frame)
    display_name = _source_display_name(project_path, request.source_asset_id)

    detection_dicts = [
        detection.to_dict(frame_width=DETECTION_FRAME_WIDTH, frame_height=DETECTION_FRAME_HEIGHT)
        for detection in detections
    ]
    runtime_meta = dict(getattr(detector, "runtime", {}) or {})
    asset = {
        "schemaVersion": 1,
        "id": asset_id,
        "projectId": request.project_id,
        "generationSetId": None,
        "type": "frame",
        "displayName": f"Person selection frame from {display_name}",
        "createdAt": created_at,
        "file": {
            "path": media_rel,
            "mimeType": "image/png",
            "width": DETECTION_FRAME_WIDTH,
            "height": DETECTION_FRAME_HEIGHT,
            "duration": None,
            "fps": None,
        },
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
        "recipe": {
            "mode": "person_detect",
            "model": getattr(detector, "model_ref", "unknown"),
            "adapter": DETECTOR_ADAPTER_ID,
            "prompt": "Detect selectable people in representative frame",
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "sourceTimestamp": timestamp,
                "detectionCount": len(detection_dicts),
                "confidence": request.confidence,
                "personDetectionActive": True,
                "detector": runtime_meta,
            },
            "rawAdapterSettings": {"sourceFrameHash": source_hash},
        },
        "lineage": {
            "parents": [request.source_asset_id],
            "sourceAssetId": request.source_asset_id,
            "sourceTimestamp": timestamp,
            "sourceFrameHash": source_hash,
            "jobId": job.get("id"),
        },
    }
    from .image_adapters import write_json  # lazy import to avoid a heavy import chain

    write_json(media_path.with_suffix(".sceneworks.json"), asset)
    write_json(project_path / "recipes" / f"{asset_id}.recipe.json", asset["recipe"])
    index_asset(project_path, asset)

    return {
        "frameAssetId": asset_id,
        "frameAsset": asset,
        "sourceAssetId": request.source_asset_id,
        "sourceTimestamp": timestamp,
        "sourceFrameHash": source_hash,
        "detections": detection_dicts,
        "detector": runtime_meta,
        "personDetectionActive": True,
    }


# ---------------------------------------------------------------------------
# Tracking (sc-1481): content-based selected-person tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameObservation:
    """Person boxes observed in one processed source frame, keyed by tracker id."""

    timestamp: float
    boxes: dict[int, tuple[NormalizedBox, float]]


class TrackerRuntime(Protocol):
    model_ref: str
    runtime: dict[str, Any]

    def observe(self, video_path: Path, *, confidence: float) -> list[FrameObservation]:
        ...


@dataclass
class TrackAssembly:
    frames: list[TrackFrame]
    target_track_id: int | None
    quality: dict[str, Any]


def _nearest_observation(observations: list[FrameObservation], timestamp: float) -> FrameObservation | None:
    if not observations:
        return None
    return min(observations, key=lambda obs: abs(obs.timestamp - timestamp))


def choose_target_track_id(
    observations: list[FrameObservation],
    selected_box: NormalizedBox,
    selected_timestamp: float,
    *,
    min_iou: float = 0.1,
) -> int | None:
    """Match the user-selected detection to a tracker identity by IoU at the
    observation nearest the selection frame. Returns None when nothing overlaps
    so tracking fails honestly instead of locking onto the wrong person."""
    observation = _nearest_observation(observations, selected_timestamp)
    if observation is None:
        return None
    best_id: int | None = None
    best_iou = min_iou
    for track_id, (box, _conf) in observation.boxes.items():
        score = box_iou(box, selected_box)
        if score >= best_iou:
            best_iou = score
            best_id = track_id
    return best_id


def assemble_track(
    observations: list[FrameObservation],
    selected_box: NormalizedBox,
    selected_timestamp: float,
    timestamps: list[float],
) -> TrackAssembly:
    """Resample a chosen tracker identity onto the requested sample cadence.

    Detected frames carry the tracker's real box/confidence; frames where the
    target is absent are recorded as ``detected=False`` and flagged, never
    fabricated into plausible motion the way the V1 placeholder did.
    """
    target_id = choose_target_track_id(observations, selected_box, selected_timestamp)
    frames: list[TrackFrame] = []
    detected_count = 0
    lost_frames: list[int] = []
    last_box = selected_box
    for index, stamp in enumerate(timestamps):
        observation = _nearest_observation(observations, stamp)
        entry = observation.boxes.get(target_id) if (observation and target_id is not None) else None
        if entry is not None:
            box, confidence = entry
            last_box = box
            flags = ["low_confidence"] if confidence < TRACK_LOW_CONFIDENCE else []
            frames.append(TrackFrame(timestamp=stamp, box=box, confidence=confidence, detected=True, flags=flags))
            detected_count += 1
        else:
            lost_frames.append(index)
            frames.append(
                TrackFrame(
                    timestamp=stamp,
                    box=last_box,
                    confidence=0.0,
                    detected=False,
                    flags=["lost_target"],
                )
            )
    quality = {
        "trackId": target_id,
        "sampledFrames": len(timestamps),
        "detectedFrames": detected_count,
        "lostFrames": lost_frames,
        "detectedRatio": round(detected_count / len(timestamps), 4) if timestamps else 0.0,
    }
    return TrackAssembly(frames=frames, target_track_id=target_id, quality=quality)


def resolve_tracker_model(settings: Any, override: str | None) -> str:
    if override:
        return override
    env_ref = os.getenv("SCENEWORKS_PERSON_TRACKER_MODEL")
    if env_ref:
        return env_ref
    return resolve_detector_model(settings, None)


def load_person_tracker(settings: Any, *, model_ref: str | None = None) -> TrackerRuntime:
    if not tracker_backend_available():
        raise RuntimeError(
            "Person tracking requires the Ultralytics backend. Install "
            "apps/worker/requirements-person.txt in the GPU worker image."
        )
    ultralytics = importlib.import_module("ultralytics")
    resolved = resolve_tracker_model(settings, model_ref)
    yolo = ultralytics.YOLO(resolved)
    tracker_cfg = os.getenv("SCENEWORKS_PERSON_TRACKER_CFG", "bytetrack.yaml")
    runtime = {
        "backend": "ultralytics",
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "tracker": tracker_cfg,
        "device": getattr(settings, "gpu_id", "cpu"),
    }
    return _UltralyticsTracker(model=yolo, model_ref=resolved, runtime=runtime, tracker_cfg=tracker_cfg)


@dataclass
class _UltralyticsTracker:
    model: Any
    model_ref: str
    runtime: dict[str, Any]
    tracker_cfg: str

    def observe(self, video_path: Path, *, confidence: float) -> list[FrameObservation]:
        results = self.model.track(
            source=str(video_path),
            classes=[PERSON_CLASS_INDEX],
            conf=confidence,
            tracker=self.tracker_cfg,
            persist=True,
            stream=True,
            verbose=False,
        )
        observations: list[FrameObservation] = []
        for index, result in enumerate(results):
            boxes = getattr(result, "boxes", None)
            timestamp = self._result_timestamp(result, index)
            entries: dict[int, tuple[NormalizedBox, float]] = {}
            if boxes is not None and getattr(boxes, "id", None) is not None:
                shape = getattr(result, "orig_shape", (DETECTION_FRAME_HEIGHT, DETECTION_FRAME_WIDTH))
                height, width = int(shape[0]), int(shape[1])
                ids = boxes.id.tolist()
                xyxy = boxes.xyxy.tolist()
                confs = boxes.conf.tolist()
                for track_id, coords, conf in zip(ids, xyxy, confs):
                    box = xyxy_to_normalized(coords[0], coords[1], coords[2], coords[3], width, height)
                    entries[int(track_id)] = (box, float(conf))
            observations.append(FrameObservation(timestamp=timestamp, boxes=entries))
        return observations

    def _result_timestamp(self, result: Any, index: int) -> float:
        speed = getattr(result, "speed", None)
        fps = None
        if isinstance(speed, dict):
            fps = speed.get("fps")
        if not fps:
            fps = PERSON_TRACK_SAMPLE_RATE_FPS
        return round(index / float(fps), 4) if fps else float(index)


# ---------------------------------------------------------------------------
# Segmentation (sc-1482): per-frame person masks
# ---------------------------------------------------------------------------


class SegmenterRuntime(Protocol):
    model_ref: str
    runtime: dict[str, Any]

    def segment(self, image: Image.Image, box: NormalizedBox) -> Image.Image:
        ...


def load_person_segmenter(settings: Any, *, model_ref: str | None = None) -> SegmenterRuntime:
    """Load a video segmentation/matting backend (SAM2 primary, MatAnyone
    accepted). Raises when unavailable so callers can degrade to box masks."""
    if not segmenter_backend_available():
        raise RuntimeError(
            "Person segmentation requires SAM2 or MatAnyone. Install "
            "apps/worker/requirements-person.txt in the GPU worker image."
        )
    if _module_available("sam2"):
        return _build_sam2_segmenter(settings, model_ref)
    return _build_matanyone_segmenter(settings, model_ref)


def _build_sam2_segmenter(settings: Any, model_ref: str | None) -> SegmenterRuntime:
    sam2 = importlib.import_module("sam2.sam2_image_predictor")
    build = importlib.import_module("sam2.build_sam")
    checkpoint = model_ref or os.getenv("SCENEWORKS_PERSON_SEGMENTER_MODEL", "sam2_hiera_large.pt")
    config = os.getenv("SCENEWORKS_PERSON_SEGMENTER_CFG", "sam2_hiera_l.yaml")
    model = build.build_sam2(config, checkpoint, device=getattr(settings, "gpu_id", "cpu"))
    predictor = sam2.SAM2ImagePredictor(model)
    runtime = {"backend": "sam2", "checkpoint": str(checkpoint), "config": config}
    return _Sam2Segmenter(predictor=predictor, model_ref=str(checkpoint), runtime=runtime)


def _build_matanyone_segmenter(settings: Any, model_ref: str | None) -> SegmenterRuntime:
    raise RuntimeError("MatAnyone segmenter integration is not yet wired; install SAM2.")


@dataclass
class _Sam2Segmenter:
    predictor: Any
    model_ref: str
    runtime: dict[str, Any]

    def segment(self, image: Image.Image, box: NormalizedBox) -> Image.Image:
        import numpy as np  # lazy: only the real backend needs numpy

        self.predictor.set_image(np.asarray(image.convert("RGB")))
        width, height = image.width, image.height
        prompt = np.array(
            [
                box.x * width,
                box.y * height,
                (box.x + box.width) * width,
                (box.y + box.height) * height,
            ]
        )
        masks, scores, _ = self.predictor.predict(box=prompt, multimask_output=False)
        mask = masks[int(np.argmax(scores))]
        return Image.fromarray((np.asarray(mask) > 0).astype("uint8") * 255, mode="L")


def mask_relative_path(track_id: str, index: int) -> str:
    return f"person-tracks/{track_id}/masks/frame_{index:06d}.png"


def segment_track(
    settings: Any,
    project_path: Path,
    source_asset_id: str,
    track_id: str,
    frames: list[TrackFrame],
    *,
    segmenter_factory: Callable[..., SegmenterRuntime] = load_person_segmenter,
    frame_loader: Callable[..., Image.Image | None] | None = None,
) -> str:
    """Generate per-frame person masks for the detected track frames and write
    them under ``person-tracks/{track_id}/masks/``. Returns the resulting
    ``maskState`` (active/generated/degraded/missing). Box-derived fallback is
    left to the replacement loader and only used in explicit degraded mode."""
    detected = [frame for frame in frames if frame.detected]
    if not detected:
        return "missing"
    if not segmenter_backend_available():
        return "degraded"

    if frame_loader is None:
        from .video_adapters import load_source_frame

        def frame_loader(timestamp: float) -> Image.Image | None:  # type: ignore[misc]
            return load_source_frame(
                project_path, source_asset_id, timestamp, DETECTION_FRAME_WIDTH, DETECTION_FRAME_HEIGHT
            )

    try:
        segmenter = segmenter_factory(settings)
    except RuntimeError:
        return "degraded"

    masks_dir = project_path / "person-tracks" / track_id / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for index, frame in enumerate(frames):
        if not frame.detected:
            continue
        image = frame_loader(frame.timestamp)
        if image is None:
            continue
        try:
            mask = segmenter.segment(image, frame.box)
        except Exception:
            continue
        rel = mask_relative_path(track_id, index + 1)
        mask.convert("L").save(project_path / rel, format="PNG")
        frame.mask = rel
        generated += 1

    if generated == 0:
        return "degraded"
    return "active" if generated == len(detected) else "generated"


# ---------------------------------------------------------------------------
# Track orchestration (sc-1481 + sc-1482)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackRequest:
    project_id: str
    source_asset_id: str
    representative_frame_asset_id: str | None
    selected_box: NormalizedBox
    selected_detection: dict[str, Any]
    selected_timestamp: float
    track_name: str
    tracker_model: str | None
    confidence: float
    segment: bool


def track_request_from_job(job: dict[str, Any], project_path: Path) -> TrackRequest:
    payload = job.get("payload") or {}
    advanced = payload.get("advanced") or {}
    detection = payload.get("detection")
    if not isinstance(detection, dict) or not isinstance(detection.get("box"), dict):
        raise ValueError("Person track job payload requires a selected detection with a box.")
    representative_id = payload.get("representativeFrameAssetId")
    selected_timestamp = _representative_timestamp(project_path, representative_id)
    return TrackRequest(
        project_id=_require(payload, "projectId"),
        source_asset_id=_require(payload, "sourceAssetId"),
        representative_frame_asset_id=representative_id if isinstance(representative_id, str) else None,
        selected_box=NormalizedBox.from_dict(detection["box"]),
        selected_detection=detection,
        selected_timestamp=selected_timestamp,
        track_name=str(payload.get("trackName") or "Selected person"),
        tracker_model=advanced.get("trackerModel") or payload.get("trackerModel"),
        confidence=safe_float(advanced.get("confidence"), DEFAULT_DETECTION_CONFIDENCE, 0.01, 1.0),
        segment=bool(advanced.get("segment", True)),
    )


def _representative_timestamp(project_path: Path, asset_id: Any) -> float:
    if not isinstance(asset_id, str):
        return 0.0
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is None:
        return 0.0
    payload = read_json(sidecar_path)
    return safe_float(payload.get("lineage", {}).get("sourceTimestamp"), 0.0, 0.0, 3600.0)


def run_person_track(
    settings: Any,
    job: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCallback | None = None,
    tracker_factory: Callable[..., TrackerRuntime] = load_person_tracker,
    segmenter_factory: Callable[..., SegmenterRuntime] = load_person_segmenter,
) -> dict[str, Any]:
    """Track the selected person through real source-video content, generate
    per-frame masks, and persist a reusable person-track sidecar (sc-1481/1482).

    Successful tracks set ``personTrackingActive: True`` and record per-frame
    timestamp/box/confidence plus model/runtime metadata and quality. A selected
    person that never matches a real track fails honestly.
    """

    def report(stage: str, value: float, message: str) -> None:
        if progress is not None:
            progress("running", stage, value, message)

    def check_cancel() -> None:
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Person tracking canceled.")

    project_path = find_project_path(settings.data_dir / "recent-projects.json", job["payload"]["projectId"])
    request = track_request_from_job(job, project_path)
    video_path = _require_source_media(project_path, request.source_asset_id)
    duration = _source_duration(project_path, request.source_asset_id)
    timestamps = sample_timestamps(duration)

    report("loading_model", 0.3, "Loading person tracker.")
    check_cancel()
    tracker = tracker_factory(settings, model_ref=request.tracker_model)

    report("tracking", 0.45, "Tracking selected person through source frames.")
    observations = tracker.observe(video_path, confidence=request.confidence)
    assembly = assemble_track(observations, request.selected_box, request.selected_timestamp, timestamps)
    if assembly.target_track_id is None or assembly.quality["detectedFrames"] == 0:
        raise RuntimeError(
            "Selected person was not found in the source video. Re-run detection or adjust the selection."
        )

    mask_state = "missing"
    if request.segment:
        report("running", 0.7, "Generating per-frame person masks.")
        check_cancel()
        mask_state = segment_track(
            settings,
            project_path,
            request.source_asset_id,
            _pending_track_id(job),
            assembly.frames,
            segmenter_factory=segmenter_factory,
        )

    report("saving", 0.85, "Saving reusable person track.")
    check_cancel()
    track_id = _pending_track_id(job)
    detected_confidences = [frame.confidence for frame in assembly.frames if frame.detected]
    average_confidence = round(sum(detected_confidences) / len(detected_confidences), 4) if detected_confidences else 0.0
    created_at = utc_now()
    runtime_meta = dict(getattr(tracker, "runtime", {}) or {})
    track = {
        "schemaVersion": 1,
        "id": track_id,
        "projectId": request.project_id,
        "name": request.track_name,
        "createdAt": created_at,
        "sourceAssetId": request.source_asset_id,
        "sourceDisplayName": _source_display_name(project_path, request.source_asset_id),
        "representativeFrameAssetId": request.representative_frame_asset_id,
        "selectedDetection": request.selected_detection,
        "frames": [frame.to_dict() for frame in assembly.frames],
        "corrections": [],
        "status": {
            "sampleRateFps": PERSON_TRACK_SAMPLE_RATE_FPS,
            "maskState": mask_state,
            "averageConfidence": average_confidence,
            "correctionState": "ready_for_box_corrections",
            "personTrackingActive": True,
            "quality": assembly.quality,
            "tracker": runtime_meta,
        },
        "recipe": {
            "mode": "person_track",
            "model": getattr(tracker, "model_ref", "unknown"),
            "adapter": TRACKER_ADAPTER_ID,
            "prompt": f"Track {request.track_name}",
            "negativePrompt": "",
            "seed": 0,
            "loras": [],
            "stylePreset": "none",
            "normalizedSettings": {
                "sampleRateFps": PERSON_TRACK_SAMPLE_RATE_FPS,
                "personDetectionActive": True,
                "personTrackingActive": True,
                "maskState": mask_state,
                "tracker": runtime_meta,
            },
            "rawAdapterSettings": {"selectedDetection": request.selected_detection},
        },
        "lineage": {
            "jobId": job.get("id"),
            "parents": [request.source_asset_id, request.representative_frame_asset_id],
            "sourceAssetId": request.source_asset_id,
        },
    }
    track_rel = f"person-tracks/{track_id}.sceneworks.person-track.json"
    from .image_adapters import write_json  # lazy import to avoid a heavy import chain

    write_json(project_path / track_rel, track)
    return {"trackId": track_id, "track": track, "path": track_rel, "personTrackingActive": True}


def _pending_track_id(job: dict[str, Any]) -> str:
    """Deterministic per-job track id so masks written during a job land under
    the same folder the sidecar references."""
    cached = job.get("_personTrackId")
    if not cached:
        cached = f"track_{uuid4().hex}"
        job["_personTrackId"] = cached
    return cached


def _require_source_media(project_path: Path, asset_id: str) -> Path:
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is None:
        raise RuntimeError(f"Source clip asset not found: {asset_id}.")
    payload = read_json(sidecar_path)
    media_path = project_path / payload.get("file", {}).get("path", "")
    if not media_path.exists():
        raise RuntimeError(f"Source clip media is missing for asset {asset_id}.")
    return media_path


def apply_track_corrections(track: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply persisted box/reject corrections to a track's frames (sc-1485).

    Returns copies of the track frames with corrections applied so replacement
    consumes corrected geometry without mutating the sidecar:

    * A corrected box overrides the tracked box and clears that frame's stored
      segmentation mask — the old mask no longer matches the new box, so the
      mask loader regenerates a box mask from the corrected box (corrected-box-
      driven mask regeneration, which the story accepts in lieu of mask paint).
    * A rejected frame is excluded: its box/mask is replaced by the nearest
      accepted frame's box (mask cleared), so the masked control clip never
      neutralizes a region using geometry the user flagged as wrong.

    Corrections are keyed by ``frameIndex``; out-of-range or malformed entries are
    ignored so a stale sidecar never breaks replacement.
    """
    frames = [dict(frame) for frame in (track.get("frames") or [])]
    if not frames:
        return frames
    by_index: dict[int, dict[str, Any]] = {}
    for correction in track.get("corrections") or []:
        if not isinstance(correction, dict):
            continue
        index = correction.get("frameIndex")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if 0 <= index < len(frames):
            by_index[index] = correction

    rejected = [False] * len(frames)
    for index, correction in by_index.items():
        frame = frames[index]
        box = correction.get("box")
        if isinstance(box, dict):
            frame["box"] = box
            frame["mask"] = None
            frame["corrected"] = True
        if correction.get("rejected"):
            rejected[index] = True
            frame["rejected"] = True

    accepted = [index for index, flag in enumerate(rejected) if not flag]
    if accepted and len(accepted) < len(frames):
        accepted_boxes = {index: frames[index].get("box") for index in accepted}
        for index, flag in enumerate(rejected):
            if not flag:
                continue
            nearest = min(accepted, key=lambda candidate: (abs(candidate - index), candidate))
            frames[index]["box"] = accepted_boxes[nearest]
            frames[index]["mask"] = None
    return frames


def load_track_masks(
    project_path: Path,
    track: dict[str, Any],
    width: int,
    height: int,
    count: int,
) -> tuple[list[Image.Image], str]:
    """Load per-frame masks for replacement, resampled to ``count`` frames.

    Persisted corrections are applied first (corrected boxes override tracked
    boxes; rejected frames borrow the nearest accepted box). Masks are then
    chosen per frame: a frame keeps its stored segmentation mask only when the
    file still exists and the box was not corrected, otherwise it falls back to a
    rectangular box mask from the (possibly corrected) box.

    Returns ``(masks, mode)`` where mode is ``"segmentation"`` (all stored masks),
    ``"degraded_box"`` (all box-derived), or ``"mixed"`` (some of each, e.g. a
    corrected frame regenerated as a box mask alongside untouched seg masks).
    """
    frames = apply_track_corrections(track)
    if not frames:
        raise RuntimeError("Person track has no frames; cannot build replacement masks.")
    indices = _resample_indices(len(frames), count)
    masks: list[Image.Image] = []
    segmentation = 0
    for index in indices:
        frame = frames[index]
        ref = frame.get("mask")
        if isinstance(ref, str) and (project_path / ref).exists():
            masks.append(Image.open(project_path / ref).convert("L").resize((width, height)))
            segmentation += 1
        else:
            masks.append(_box_mask(frame.get("box") or {}, width, height))
    if segmentation == len(indices):
        mode = "segmentation"
    elif segmentation == 0:
        mode = "degraded_box"
    else:
        mode = "mixed"
    return masks, mode


def _resample_indices(total: int, count: int) -> list[int]:
    if count <= 1 or total <= 1:
        return [0] * max(count, 1)
    return [min(total - 1, round((total - 1) * index / (count - 1))) for index in range(count)]


def _box_mask(box: dict[str, Any], width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    if box:
        pad_x, pad_y = int(width * 0.03), int(height * 0.03)
        left = max(0, int(safe_float(box.get("x"), 0.0, 0.0, 1.0) * width) - pad_x)
        top = max(0, int(safe_float(box.get("y"), 0.0, 0.0, 1.0) * height) - pad_y)
        right = min(width, int((safe_float(box.get("x"), 0.0, 0.0, 1.0) + safe_float(box.get("width"), 0.0, 0.0, 1.0)) * width) + pad_x)
        bottom = min(height, int((safe_float(box.get("y"), 0.0, 0.0, 1.0) + safe_float(box.get("height"), 0.0, 0.0, 1.0)) * height) + pad_y)
        ImageDraw.Draw(mask).rectangle((left, top, right, bottom), fill=255)
    return mask
