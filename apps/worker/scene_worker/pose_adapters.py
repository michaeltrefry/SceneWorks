"""DWPose whole-body keypoint detection for the Pose Library (epic 2282, sc-2285).

Photo -> normalized whole-body keypoints (OpenPose-18 body + 21x2 hands + 68 face
+ per-keypoint confidence + per-person bbox) + source aspect, rendered into a
skeleton preview via the production ``openpose_skeleton.draw_wholebody`` (the same
control the Z-Image Fun-Controlnet-Union pose head was trained on — sc-2257/PR#376).

Design mirrors ``person_adapters``: the heavy backend (rtmlib / onnxruntime) is
imported lazily inside the model seam so this module stays importable on a host
without it (the ``pose_detect`` capability is then simply not advertised). The
pure keypoint conversion + geometry is plain Python/numpy and unit-testable
without the onnx weights.

Detector: rtmlib (Apache-2.0) RTMW whole-body on onnxruntime — the same arm64
onnxruntime stack already shipped for InstantID's antelopev2. ``mode=performance``
= yolox_m + rtmw-dw-x-l (cocktail14, 384x288). The spike (sc-2283) validated this
on the M5 Max via CoreML (~33ms/image) and confirmed full DWPose never degrades
the Z-Image pose lock and fixes the high-scale drift.

The job returns one pose candidate per detected person; persisting candidates into
the global pose store (asset records) is the Create-tab/store work (sc-2287/2284).
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

ProgressCallback = Callable[[str, str, float, str], None]
CancelCallback = Callable[[], bool]

# Default per-keypoint confidence floor for rendering. NOTE: rtmlib's RTMW emits
# SimCC scores that are NOT in [0, 1] (good keypoints observed ~4-8 in the spike),
# so this is a render/threshold knob, not a probability. Raw scores are preserved
# in the output so the store can normalize/threshold per detector later (sc-2285
# spike gotcha).
DEFAULT_POSE_MIN_CONF = 0.3
DETECTOR_MODE = os.environ.get("SCENEWORKS_DWPOSE_MODE", "performance")

# COCO-WholeBody-133 (rtmlib raw, to_openpose=False) -> SceneWorks OpenPose-18 body.
# WholeBody order: 0-16 COCO body, 17-22 feet, 23-90 face(68), 91-111 left hand,
# 112-132 right hand. COCO body: 0 nose,1 l_eye,2 r_eye,3 l_ear,4 r_ear,5 l_sho,
# 6 r_sho,7 l_elb,8 r_elb,9 l_wri,10 r_wri,11 l_hip,12 r_hip,13 l_kne,14 r_kne,
# 15 l_ank,16 r_ank. OpenPose-18 inserts neck(1) = midpoint of shoulders.
COCO_TO_OPENPOSE = {
    0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16,
    11: 11, 12: 13, 13: 15, 14: 2, 15: 1, 16: 4, 17: 3,
}
FACE_SLICE = slice(23, 91)        # 68 points
LHAND_SLICE = slice(91, 112)      # 21 points
RHAND_SLICE = slice(112, 133)     # 21 points


class PoseDetectError(RuntimeError):
    """Raised when pose detection cannot proceed (missing backend / no input)."""


# ---------------------------------------------------------------------------
# Backend availability (advertised as a worker capability only when installed)
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def pose_detector_backend_available() -> bool:
    """True when the DWPose backend (rtmlib + onnxruntime) can be imported."""
    return _module_available("rtmlib") and _module_available("onnxruntime")


def _require_pose_extras() -> None:
    missing = [m for m in ("rtmlib", "onnxruntime") if not _module_available(m)]
    if missing:
        raise PoseDetectError(
            "DWPose detection needs " + " + ".join(missing) + ". Install "
            "apps/worker/requirements-pose.txt (rtmlib, onnxruntime)."
        )


# ---------------------------------------------------------------------------
# Keypoint conversion (pure; numpy only)
# ---------------------------------------------------------------------------


def _pt(kps, sc, i: int, w: int, h: int) -> list:
    return [float(kps[i, 0]) / w, float(kps[i, 1]) / h, float(sc[i])]


def _seq(kps, sc, sl: slice, w: int, h: int) -> list:
    return [_pt(kps, sc, i, w, h) for i in range(sl.start, sl.stop)]


def wholebody_to_openpose(kps, sc, w: int, h: int) -> dict:
    """Convert one person's (133,2) keypoints + (133,) scores into the SceneWorks
    pose record: body18 + hands[left21,right21] + face68, each [x,y,conf] in [0,1]."""
    body: list = [None] * 18
    for op_idx, coco_idx in COCO_TO_OPENPOSE.items():
        body[op_idx] = _pt(kps, sc, coco_idx, w, h)
    ls, rs = kps[5], kps[6]  # shoulders -> neck
    body[1] = [
        float((ls[0] + rs[0]) / 2) / w,
        float((ls[1] + rs[1]) / 2) / h,
        float(min(sc[5], sc[6])),
    ]
    return {
        "keypoints": body,
        "hands": [_seq(kps, sc, LHAND_SLICE, w, h), _seq(kps, sc, RHAND_SLICE, w, h)],
        "face": _seq(kps, sc, FACE_SLICE, w, h),
    }


def squareify(rec: dict, w: int, h: int) -> dict:
    """Re-normalize a source-aspect pose into a centered ``max(w, h)`` SQUARE — pad
    the short axis, never crop — so the stored pose is aspect-canonical. A later
    square aspect-fit at generation (``openpose_skeleton.square_fit``) then preserves
    the captured human proportions at any output aspect, instead of stretching x/y
    independently (epic 2282). Confidence is carried through unchanged. Operates on
    the body18 + hands[21,21] + face68 ``[x, y, conf]`` record from
    ``wholebody_to_openpose``."""
    side = float(max(w, h))
    ox = (side - w) / 2.0
    oy = (side - h) / 2.0

    def _sq(p):
        return None if p is None else [(p[0] * w + ox) / side, (p[1] * h + oy) / side, p[2]]

    return {
        "keypoints": [_sq(p) for p in rec["keypoints"]],
        "hands": [[_sq(p) for p in rec["hands"][0]], [_sq(p) for p in rec["hands"][1]]],
        "face": [_sq(p) for p in rec["face"]],
    }


def _thresholded(group: list, min_conf: float) -> list:
    """Render-ready copy: drop (None) points below the confidence floor."""
    out = []
    for p in group:
        out.append(None if (p is None or p[2] < min_conf) else (p[0], p[1]))
    return out


def _bbox(*groups: list, min_conf: float):
    xs, ys = [], []
    for g in groups:
        for p in g:
            if p is not None and p[2] >= min_conf:
                xs.append(p[0])
                ys.append(p[1])
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _mean_conf(group: list) -> float:
    cs = [p[2] for p in group if p is not None]
    return round(sum(cs) / len(cs), 3) if cs else 0.0


def _facing(body: list, min_conf: float) -> str:
    def ok(i):
        return body[i] is not None and body[i][2] >= min_conf
    nose, r_eye, l_eye, r_ear, l_ear = ok(0), ok(14), ok(15), ok(16), ok(17)
    if not nose and not r_eye and not l_eye:
        return "back"
    if r_ear and l_ear:
        return "front"
    if r_ear != l_ear:
        return "profile"
    return "front"


# ---------------------------------------------------------------------------
# Detector model seam
# ---------------------------------------------------------------------------


@dataclass
class PoseDetectorRuntime:
    model: Any
    device: str
    detector_id: str


_DETECTOR_CACHE: dict[str, PoseDetectorRuntime] = {}


def _onnx_device(settings: Any) -> str:
    gpu_id = str(getattr(settings, "gpu_id", "") or "").lower()
    # onnxruntime runs DWPose on CoreML (Apple Silicon) or CPU; never torch-CUDA here.
    return "cpu" if gpu_id in ("", "cpu") else "mps"


def load_pose_detector(settings: Any, *, device: str | None = None) -> PoseDetectorRuntime:
    """Load (and cache) the rtmlib whole-body detector.

    Tries the requested onnxruntime provider (CoreML for ``mps``) and falls back to
    CPU if it can't initialise. Weights come from rtmlib's ``performance`` preset;
    set ``SCENEWORKS_DWPOSE_DET``/``SCENEWORKS_DWPOSE_POSE`` to pinned local onnx
    paths to run fully offline (no openmmlab fetch)."""
    _require_pose_extras()
    from rtmlib import Wholebody

    want = device or _onnx_device(settings)
    if want in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[want]

    kwargs: dict[str, Any] = {"mode": DETECTOR_MODE, "backend": "onnxruntime"}
    det_path = os.environ.get("SCENEWORKS_DWPOSE_DET")
    pose_path = os.environ.get("SCENEWORKS_DWPOSE_POSE")
    if det_path and pose_path:
        # Pinned weights: performance preset input sizes (yolox 640, rtmw 288x384).
        kwargs = {
            "det": det_path,
            "det_input_size": (640, 640),
            "pose": pose_path,
            "pose_input_size": (288, 384),
            "backend": "onnxruntime",
        }

    try:
        model = Wholebody(device=want, **kwargs)
        used = want
    except Exception:  # noqa: BLE001 - any provider init failure -> CPU
        model = Wholebody(device="cpu", **kwargs)
        used = "cpu"

    runtime = PoseDetectorRuntime(
        model=model, device=used, detector_id=f"rtmlib/wholebody-{DETECTOR_MODE}"
    )
    _DETECTOR_CACHE[used] = runtime
    return runtime


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _normalize_sources(payload: dict) -> list[dict]:
    sources = payload.get("sources")
    if isinstance(sources, list) and sources:
        return [s for s in sources if isinstance(s, dict)]
    # single-source convenience form
    if payload.get("path"):
        return [{
            "path": payload.get("path"),
            "assetId": payload.get("sourceAssetId"),
            "displayName": payload.get("displayName"),
        }]
    return []


def _resolve_source_path(src: dict, project_path) -> str | None:
    """Resolve a source image to a filesystem path the detector can read.

    Prefers an explicit absolute ``path`` (the spike/tests pass one); otherwise
    resolves ``assetId`` against the project — the Create tab sends asset ids, not
    paths, since the browser never knows the on-disk location. Falls back to a
    project-relative ``path``. Returns ``None`` when nothing resolves so the source
    is reported as unreadable rather than crashing the whole job.
    """
    raw = src.get("path")
    if raw and os.path.isabs(raw) and os.path.exists(raw):
        return raw
    asset_id = src.get("assetId")
    if asset_id and project_path is not None:
        try:
            from sceneworks_shared import load_asset_with_media

            _record, media_path = load_asset_with_media(project_path, asset_id)
            return str(media_path)
        except Exception:  # noqa: BLE001 - missing asset -> unreadable below
            return None
    if raw and project_path is not None:
        candidate = Path(project_path) / raw
        if candidate.exists():
            return str(candidate)
    return raw


def _cleanup_temp_sources(paths: list[str], uploads_root: Path) -> None:
    """Delete pose-source temp uploads after detection. File-Upload sources are
    transient — never workspace assets (epic 2282). Guarded to the pose-uploads
    cache so a project asset resolved by id can never be removed."""
    for raw in paths:
        try:
            resolved = Path(raw).resolve()
            if uploads_root in resolved.parents:
                resolved.unlink(missing_ok=True)
        except OSError:
            pass


def run_pose_detect(
    settings: Any,
    job: dict[str, Any],
    *,
    progress: ProgressCallback | None = None,
    cancel_requested: CancelCallback | None = None,
    detector_factory: Callable[..., PoseDetectorRuntime] = load_pose_detector,
) -> dict[str, Any]:
    """Run DWPose on each source image and return whole-body pose candidates.

    payload: ``{"sources": [{"path", "assetId"?, "displayName"?}], "minConf"?}``
    result:  ``{"sources": [{source meta + poses[]}], "detector", "poseDetectionActive"}``
    Skeleton previews are written to ``<data_dir>/cache/pose_detect/<job_id>/``.
    """
    import cv2
    import numpy as np  # noqa: F401  (cv2/np used; numpy ensures conversion dtypes)

    from scene_worker.openpose_skeleton import draw_wholebody

    payload = job.get("payload") or {}
    sources = _normalize_sources(payload)
    if not sources:
        raise PoseDetectError("No source images supplied for pose detection.")
    try:
        min_conf = float(payload.get("minConf"))
    except (TypeError, ValueError):
        min_conf = DEFAULT_POSE_MIN_CONF

    # Sources may be asset ids (Create tab) rather than paths; resolve them against
    # the originating project (mirrors person_detect). Best-effort: a missing project
    # leaves project_path None and per-source resolution falls back to raw paths.
    project_id = payload.get("projectId") or job.get("projectId")
    project_path = None
    if project_id:
        try:
            from sceneworks_shared import find_project_path

            project_path = find_project_path(
                Path(getattr(settings, "data_dir", ".")) / "recent-projects.json",
                project_id,
            )
        except Exception:  # noqa: BLE001 - resolve each source individually instead
            project_path = None

    job_id = str(job.get("id") or uuid4().hex)
    out_dir = Path(getattr(settings, "data_dir", ".")) / "cache" / "pose_detect" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def _p(stage: str, value: float, message: str) -> None:
        if progress is not None:
            progress("running", stage, value, message)

    _p("downloading", 0.08, "Loading DWPose detector.")
    runtime = detector_factory(settings)

    out_sources: list[dict] = []
    uploads_root = (Path(getattr(settings, "data_dir", ".")) / "cache" / "pose-uploads").resolve()
    temp_sources: list[str] = []
    total = len(sources)
    for si, src in enumerate(sources):
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Pose detection canceled.")
        path = _resolve_source_path(src, project_path)
        if src.get("temp") and path:
            temp_sources.append(path)
        img = cv2.imread(path) if path else None
        if img is None:
            out_sources.append({
                "sourceAssetId": src.get("assetId"),
                "sourcePath": path,
                "error": "unreadable image",
                "poses": [],
            })
            continue
        h, w = img.shape[:2]
        keypoints, scores = runtime.model(img)
        n = 0 if keypoints is None else len(keypoints)

        ordered = []
        for i in range(n):
            rec = wholebody_to_openpose(keypoints[i], scores[i], w, h)
            # Store square-canonical (pad short axis): proportions then survive a
            # square aspect-fit at any generation resolution (epic 2282).
            rec = squareify(rec, w, h)
            bbox = _bbox(rec["keypoints"], rec["hands"][0], rec["hands"][1], rec["face"], min_conf=min_conf)
            area = 0.0 if bbox is None else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            ordered.append((area, rec, bbox))
        ordered.sort(key=lambda t: -t[0])  # largest person first

        stem = Path(path).stem
        poses = []
        for person_index, (_area, rec, bbox) in enumerate(ordered):
            body_t = _thresholded(rec["keypoints"], min_conf)
            hands_t = [_thresholded(rec["hands"][0], min_conf), _thresholded(rec["hands"][1], min_conf)]
            face_t = _thresholded(rec["face"], min_conf)
            # Render the (already square-canonical) skeleton on a SQUARE canvas so
            # the stored preview/thumbnail is square; square_fit maps it 1:1.
            side = max(w, h)
            stick = max(6, round(side * 0.012))
            skel = draw_wholebody(side, side, body_t, hands_t, face_t, stickwidth=stick)
            preview = out_dir / f"{stem}_p{person_index}_skel.png"
            cv2.imwrite(str(preview), cv2.cvtColor(skel, cv2.COLOR_RGB2BGR))
            poses.append({
                "personIndex": person_index,
                "bbox": bbox,
                "facing": _facing(rec["keypoints"], min_conf),
                "meanConf": {
                    "body": _mean_conf(rec["keypoints"]),
                    "hands": round((_mean_conf(rec["hands"][0]) + _mean_conf(rec["hands"][1])) / 2, 3),
                    "face": _mean_conf(rec["face"]),
                },
                "keypoints": rec["keypoints"],
                "hands": rec["hands"],
                "face": rec["face"],
                "skeletonPreview": str(preview),
            })

        out_sources.append({
            "sourceAssetId": src.get("assetId"),
            "sourcePath": path,
            "displayName": src.get("displayName") or stem,
            "sourceWidth": w,
            "sourceHeight": h,
            "sourceAspect": round(w / h, 4),
            "poses": poses,
        })
        _p("running", 0.08 + 0.9 * (si + 1) / total,
           f"Detected {len(poses)} pose(s) in image {si + 1}/{total}.")

    # File-Upload sources are transient — delete them now detection is done (the
    # startup sweep backstops canceled/failed jobs). epic 2282.
    _cleanup_temp_sources(temp_sources, uploads_root)

    return {
        "sources": out_sources,
        "detector": {"id": runtime.detector_id, "device": runtime.device, "minConf": min_conf},
        "poseDetectionActive": True,
    }
