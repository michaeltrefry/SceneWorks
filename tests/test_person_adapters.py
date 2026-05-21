"""Tests for real person detection/tracking/segmentation (epic sc-1090).

These exercise the orchestration and pure helpers on the host with fake,
content-reading model backends. The fakes inspect pixels, so the tests fail
against the old procedural placeholders (hard-coded candidate boxes / synthetic
x-drift) that ignored frame content. Real model execution is validated in the
GPU worker container; see sc-1486.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from sceneworks_shared import index_asset, write_json
from scene_worker import person_adapters as pa
from scene_worker.person_adapters import (
    NormalizedBox,
    PersonDetection,
    box_iou,
    sample_timestamps,
    select_target_index,
    xyxy_to_normalized,
)
from scene_worker.video_adapters import (
    LtxPipelinesResources,
    LtxPipelinesVideoAdapter,
    build_video_asset_sidecar,
    person_track_masks,
    video_request_from_job,
)


# ---------------------------------------------------------------------------
# Synthetic CI fixture clip (no committed binary): a bright "person" rectangle
# that translates left-to-right across a dark background.
# ---------------------------------------------------------------------------

CLIP_W, CLIP_H = 1280, 720
BACKGROUND = (18, 17, 15)


def _person_frame(progress: float, *, present: bool = True) -> Image.Image:
    frame = Image.new("RGB", (CLIP_W, CLIP_H), BACKGROUND)
    if present:
        box_w, box_h = 200, 420
        left = int((CLIP_W - box_w) * progress)
        top = (CLIP_H - box_h) // 2
        ImageDraw.Draw(frame).rectangle((left, top, left + box_w, top + box_h), fill=(235, 220, 205))
    return frame


def _synthetic_clip(frames: int = 8, *, present: bool = True) -> list[Image.Image]:
    if frames <= 1:
        return [_person_frame(0.0, present=present)]
    return [_person_frame(index / (frames - 1), present=present) for index in range(frames)]


def _write_clip(path: Path, frames: list[Image.Image]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head, *tail = frames
    head.save(path, format="WEBP", save_all=True, append_images=tail, duration=500, loop=0)


def _make_project(tmp_path: Path, frames: list[Image.Image], duration: float = 4.0) -> SimpleNamespace:
    project_path = tmp_path / "project"
    (project_path / "assets" / "videos").mkdir(parents=True, exist_ok=True)
    source_id = "asset_source_clip"
    media_rel = "assets/videos/source.webp"
    _write_clip(project_path / media_rel, frames)
    source_asset = {
        "schemaVersion": 1,
        "id": source_id,
        "projectId": "proj_1",
        "type": "video",
        "displayName": "Synthetic source",
        "createdAt": "2026-05-21T00:00:00Z",
        "file": {"path": media_rel, "mimeType": "image/webp", "width": CLIP_W, "height": CLIP_H, "duration": duration, "fps": 2},
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
        "recipe": {},
        "lineage": {},
    }
    write_json((project_path / media_rel).with_suffix(".sceneworks.json"), source_asset)
    index_asset(project_path, source_asset)
    (tmp_path / "recent-projects.json").write_text(
        f'[{{"id": "proj_1", "path": "{project_path.as_posix()}"}}]', encoding="utf-8"
    )
    return SimpleNamespace(data_dir=tmp_path, gpu_id="cpu", source_asset_id=source_id)


def _bright_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    mask = image.convert("L").point(lambda value: 255 if value > 80 else 0)
    return mask.getbbox()


class FakeContentDetector:
    """A detector that reads pixels: returns one box around the bright region,
    nothing when the frame is empty. Proves the pipeline is content-dependent."""

    model_ref = "fake-content-detector"

    def __init__(self) -> None:
        self.runtime = {"backend": "fake", "device": "cpu"}

    def detect(self, image: Image.Image, *, confidence: float) -> list[PersonDetection]:
        bbox = _bright_bbox(image)
        if bbox is None:
            return []
        box = xyxy_to_normalized(*bbox, image.width, image.height)
        return [PersonDetection(id="person_1", label="Person 1", box=box, confidence=0.93)]


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


def test_box_iou_identical_is_one_and_disjoint_is_zero():
    a = NormalizedBox(0.1, 0.1, 0.2, 0.2)
    assert box_iou(a, a) == pytest.approx(1.0)
    b = NormalizedBox(0.7, 0.7, 0.2, 0.2)
    assert box_iou(a, b) == 0.0


def test_xyxy_to_normalized_orders_and_clamps():
    box = xyxy_to_normalized(640, 360, 320, 180, 1280, 720)
    assert box.x == pytest.approx(0.25)
    assert box.y == pytest.approx(0.25)
    assert box.width == pytest.approx(0.25)
    assert box.height == pytest.approx(0.25)


def test_sample_timestamps_span_clip_and_clamp_count():
    stamps = sample_timestamps(4.0)
    assert stamps[0] == 0.0
    assert stamps[-1] == pytest.approx(4.0)
    assert pa.PERSON_TRACK_MIN_SAMPLES <= len(stamps) <= pa.PERSON_TRACK_MAX_SAMPLES
    assert len(sample_timestamps(0.0)) == 1


def test_select_target_index_returns_none_when_no_overlap():
    detections = [PersonDetection("person_1", "Person 1", NormalizedBox(0.0, 0.0, 0.1, 0.1), 0.9)]
    assert select_target_index(detections, NormalizedBox(0.8, 0.8, 0.1, 0.1)) is None
    assert select_target_index(detections, NormalizedBox(0.0, 0.0, 0.12, 0.12)) == 0


# ---------------------------------------------------------------------------
# Detection orchestration tests (sc-1480)
# ---------------------------------------------------------------------------


def _detect_job(settings: SimpleNamespace) -> dict:
    return {
        "id": "job_detect_1",
        "payload": {"projectId": "proj_1", "sourceAssetId": settings.source_asset_id, "sourceTimestamp": 0.0},
    }


def test_detection_is_content_dependent_and_marks_active(tmp_path):
    settings = _make_project(tmp_path, _synthetic_clip(present=True))
    result = pa.run_person_detect(
        settings,
        _detect_job(settings),
        detector_factory=lambda *_args, **_kwargs: FakeContentDetector(),
    )
    assert result["personDetectionActive"] is True
    assert len(result["detections"]) == 1
    assert result["detector"]["backend"] == "fake"
    assert result["sourceFrameHash"]
    detection = result["detections"][0]
    assert detection["box"]["width"] > 0 and detection["box"]["height"] > 0
    # frame asset persisted + indexed
    frame_path = Path(settings.data_dir) / "project" / result["frameAsset"]["file"]["path"]
    assert frame_path.exists()
    with sqlite3.connect(Path(settings.data_dir) / "project" / "project.db") as connection:
        row = connection.execute("select type from assets where id = ?", (result["frameAssetId"],)).fetchone()
    assert row is not None and row[0] == "frame"


def test_detection_returns_no_candidates_for_empty_frame(tmp_path):
    settings = _make_project(tmp_path, _synthetic_clip(present=False))
    result = pa.run_person_detect(
        settings,
        _detect_job(settings),
        detector_factory=lambda *_args, **_kwargs: FakeContentDetector(),
    )
    # No fake boxes for a person-free frame — the old template detector always returned 3.
    assert result["detections"] == []
    assert result["personDetectionActive"] is True


def test_detection_never_marks_inactive_on_success(tmp_path):
    settings = _make_project(tmp_path, _synthetic_clip(present=True))
    result = pa.run_person_detect(
        settings,
        _detect_job(settings),
        detector_factory=lambda *_args, **_kwargs: FakeContentDetector(),
    )
    normalized = result["frameAsset"]["recipe"]["normalizedSettings"]
    assert normalized["personDetectionActive"] is True
    assert result["frameAsset"]["recipe"]["adapter"] == pa.DETECTOR_ADAPTER_ID


def test_load_person_detector_errors_clearly_without_backend(monkeypatch):
    monkeypatch.setattr(pa, "detector_backend_available", lambda: False)
    with pytest.raises(RuntimeError, match="Ultralytics backend"):
        pa.load_person_detector(SimpleNamespace(data_dir=Path("."), gpu_id="cpu"))


# ---------------------------------------------------------------------------
# Tracking tests (sc-1481)
# ---------------------------------------------------------------------------


class FakeContentTracker:
    """Reads the source video and tracks the bright region as id 1, dropping it
    when it leaves the frame — so output follows real pixel motion, not drift."""

    model_ref = "fake-content-tracker"

    def __init__(self) -> None:
        self.runtime = {"backend": "fake", "tracker": "fake-bytetrack", "device": "cpu"}

    def observe(self, video_path: Path, *, confidence: float) -> list[pa.FrameObservation]:
        image = Image.open(video_path)
        total = getattr(image, "n_frames", 1)
        observations = []
        for index in range(total):
            image.seek(index)
            frame = image.convert("RGB")
            bbox = _bright_bbox(frame)
            entries = {}
            if bbox is not None:
                entries[1] = (xyxy_to_normalized(*bbox, frame.width, frame.height), 0.9)
            observations.append(pa.FrameObservation(timestamp=round(index / 2.0, 4), boxes=entries))
        return observations


class FakeEllipseSegmenter:
    model_ref = "fake-ellipse-segmenter"
    runtime = {"backend": "fake-seg"}

    def segment(self, image: Image.Image, box: NormalizedBox) -> Image.Image:
        mask = Image.new("L", (image.width, image.height), 0)
        left, top = box.x * image.width, box.y * image.height
        right, bottom = (box.x + box.width) * image.width, (box.y + box.height) * image.height
        ImageDraw.Draw(mask).ellipse((left, top, right, bottom), fill=255)
        return mask


def _selected_box_from_first_frame(frames: list[Image.Image]) -> NormalizedBox:
    bbox = _bright_bbox(frames[0])
    assert bbox is not None
    return xyxy_to_normalized(*bbox, frames[0].width, frames[0].height)


def test_choose_target_returns_none_without_overlap():
    obs = [pa.FrameObservation(0.0, {1: (NormalizedBox(0.0, 0.0, 0.1, 0.8), 0.9)})]
    assert pa.choose_target_track_id(obs, NormalizedBox(0.9, 0.0, 0.05, 0.05), 0.0) is None
    assert pa.choose_target_track_id(obs, NormalizedBox(0.0, 0.0, 0.12, 0.8), 0.0) == 1


def test_assemble_track_marks_lost_frames_honestly():
    selected = NormalizedBox(0.0, 0.1, 0.15, 0.6)
    observations = [
        pa.FrameObservation(0.0, {1: (NormalizedBox(0.0, 0.1, 0.15, 0.6), 0.9)}),
        pa.FrameObservation(1.0, {}),  # target lost
        pa.FrameObservation(2.0, {1: (NormalizedBox(0.4, 0.1, 0.15, 0.6), 0.8)}),
    ]
    assembly = pa.assemble_track(observations, selected, 0.0, [0.0, 1.0, 2.0])
    assert assembly.target_track_id == 1
    assert [frame.detected for frame in assembly.frames] == [True, False, True]
    assert "lost_target" in assembly.frames[1].flags
    assert assembly.frames[1].confidence == 0.0  # not a synthetic box
    assert assembly.quality["detectedFrames"] == 2


def _track_job(settings, frames) -> tuple[dict, NormalizedBox]:
    selected = _selected_box_from_first_frame(frames)
    job = {
        "id": "job_track_1",
        "payload": {
            "projectId": "proj_1",
            "sourceAssetId": settings.source_asset_id,
            "trackName": "Hero",
            "detection": {"id": "person_1", "label": "Person 1", "box": selected.to_dict(), "confidence": 0.93},
        },
    }
    return job, selected


def test_tracking_follows_real_motion_and_marks_active(tmp_path, monkeypatch):
    frames = _synthetic_clip(frames=8, present=True)
    settings = _make_project(tmp_path, frames)
    job, _selected = _track_job(settings, frames)
    monkeypatch.setattr(pa, "segmenter_backend_available", lambda: True)

    result = pa.run_person_track(
        settings,
        job,
        tracker_factory=lambda *_a, **_k: FakeContentTracker(),
        segmenter_factory=lambda *_a, **_k: FakeEllipseSegmenter(),
    )
    track = result["track"]
    assert result["personTrackingActive"] is True
    assert track["status"]["personTrackingActive"] is True
    detected = [frame for frame in track["frames"] if frame["detected"]]
    assert len(detected) >= 4
    # Boxes follow the rightward motion in the pixels, not a fixed synthetic drift.
    assert detected[-1]["box"]["x"] > detected[0]["box"]["x"] + 0.2
    assert track["status"]["maskState"] == "active"
    assert track["status"]["averageConfidence"] > 0


def test_tracking_fails_honestly_when_selection_absent(tmp_path):
    frames = _synthetic_clip(frames=8, present=True)
    settings = _make_project(tmp_path, frames)
    job = {
        "id": "job_track_2",
        "payload": {
            "projectId": "proj_1",
            "sourceAssetId": settings.source_asset_id,
            "trackName": "Ghost",
            # A selection box in a region the person never occupies at t=0.
            "detection": {"id": "person_9", "box": {"x": 0.85, "y": 0.0, "width": 0.05, "height": 0.05}},
        },
    }
    with pytest.raises(RuntimeError, match="not found in the source video"):
        pa.run_person_track(
            settings,
            job,
            tracker_factory=lambda *_a, **_k: FakeContentTracker(),
        )


# ---------------------------------------------------------------------------
# Segmentation + mask-loading tests (sc-1482)
# ---------------------------------------------------------------------------


def test_segment_track_writes_non_rectangular_masks(tmp_path, monkeypatch):
    frames_imgs = _synthetic_clip(frames=4, present=True)
    settings = _make_project(tmp_path, frames_imgs)
    monkeypatch.setattr(pa, "segmenter_backend_available", lambda: True)
    selected = _selected_box_from_first_frame(frames_imgs)
    track_frames = [pa.TrackFrame(timestamp=float(i), box=selected, confidence=0.9, detected=True) for i in range(4)]
    project_path = Path(settings.data_dir) / "project"

    state = pa.segment_track(
        settings,
        project_path,
        settings.source_asset_id,
        "track_test",
        track_frames,
        segmenter_factory=lambda *_a, **_k: FakeEllipseSegmenter(),
        frame_loader=lambda timestamp: frames_imgs[0],
    )
    assert state == "active"
    assert all(frame.mask for frame in track_frames)
    mask_img = Image.open(project_path / track_frames[0].mask).convert("L")
    bbox = mask_img.getbbox()
    nonzero = sum(mask_img.histogram()[1:])
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    assert nonzero < bbox_area * 0.95  # an ellipse fills < its bounding rectangle


def test_segment_track_degrades_without_backend(tmp_path, monkeypatch):
    settings = _make_project(tmp_path, _synthetic_clip(frames=2, present=True))
    monkeypatch.setattr(pa, "segmenter_backend_available", lambda: False)
    track_frames = [pa.TrackFrame(0.0, NormalizedBox(0.1, 0.1, 0.2, 0.5), 0.9, detected=True)]
    state = pa.segment_track(settings, Path(settings.data_dir) / "project", settings.source_asset_id, "t", track_frames)
    assert state == "degraded"


def test_load_track_masks_prefers_segmentation_then_degrades(tmp_path):
    project_path = tmp_path / "project"
    masks_dir = project_path / "person-tracks" / "track_x" / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    rel = "person-tracks/track_x/masks/frame_000001.png"
    Image.new("L", (64, 64), 255).save(project_path / rel)
    track_with_masks = {"frames": [{"box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.5}, "mask": rel}]}
    masks, mode = pa.load_track_masks(project_path, track_with_masks, 32, 32, 1)
    assert mode == "segmentation"
    assert masks[0].size == (32, 32)

    track_no_masks = {"frames": [{"box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.5}, "mask": None}]}
    masks, mode = pa.load_track_masks(project_path, track_no_masks, 32, 32, 1)
    assert mode == "degraded_box"
    assert masks[0].getbbox() is not None


# ---------------------------------------------------------------------------
# Active LTX masked-control replacement tests (sc-1483 / sc-1486)
# ---------------------------------------------------------------------------


def _write_person_track(project_path: Path, track_id: str, *, with_masks: bool) -> None:
    masks_dir = project_path / "person-tracks" / track_id / "masks"
    frames = []
    for index in range(3):
        box = {"x": 0.2 + index * 0.05, "y": 0.15, "width": 0.2, "height": 0.6}
        mask_rel = None
        if with_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_rel = pa.mask_relative_path(track_id, index + 1)
            mask_img = Image.new("L", (CLIP_W, CLIP_H), 0)
            ImageDraw.Draw(mask_img).ellipse((200, 100, 400, 520), fill=255)
            mask_img.save(project_path / mask_rel)
        frames.append({"timestamp": float(index), "box": box, "confidence": 0.9, "detected": True, "mask": mask_rel})
    track = {
        "schemaVersion": 1,
        "id": track_id,
        "projectId": "proj_1",
        "name": "Hero",
        "sourceAssetId": "asset_source_clip",
        "selectedDetection": {"id": "person_1", "box": frames[0]["box"], "confidence": 0.9},
        "frames": frames,
        "status": {
            "maskState": "active" if with_masks else "degraded",
            "personTrackingActive": True,
            "averageConfidence": 0.9,
        },
    }
    write_json(project_path / "person-tracks" / f"{track_id}.sceneworks.person-track.json", track)


def _write_character(project_path: Path, character_id: str) -> None:
    ref_id = "asset_char_ref_1"
    ref_rel = "assets/images/char_ref.png"
    (project_path / "assets" / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), (200, 120, 90)).save(project_path / ref_rel)
    ref_asset = {
        "schemaVersion": 1,
        "id": ref_id,
        "projectId": "proj_1",
        "type": "image",
        "displayName": "Character ref",
        "createdAt": "2026-05-21T00:00:00Z",
        "file": {"path": ref_rel, "mimeType": "image/png", "width": 512, "height": 512, "duration": None, "fps": None},
        "status": {"favorite": False, "rating": 0, "rejected": False, "trashed": False},
        "recipe": {},
        "lineage": {},
    }
    write_json((project_path / ref_rel).with_suffix(".sceneworks.json"), ref_asset)
    index_asset(project_path, ref_asset)
    character = {
        "schemaVersion": 1,
        "id": character_id,
        "name": "Mira",
        "references": [{"assetId": ref_id, "approved": True}],
        "looks": [],
    }
    write_json(project_path / "characters" / f"{character_id}.sceneworks.character.json", character)


def _replacement_job(settings, *, mock: bool = False) -> dict:
    advanced = {"mockNativeInference": True} if mock else {}
    return {
        "id": "job_replace_1",
        "payload": {
            "projectId": "proj_1",
            "mode": "replace_person",
            "model": "ltx_2_3",
            "prompt": "replace the hero",
            "duration": 1,
            "fps": 8,
            "width": 512,
            "height": 320,
            "sourceClipAssetId": settings.source_asset_id,
            "personTrackId": "track_repl",
            "characterId": "char_1",
            "replacementMode": "full_person_keep_outfit",
            "advanced": advanced,
        },
    }


def _seed_replacement_project(tmp_path, *, with_masks: bool = True) -> SimpleNamespace:
    settings = _make_project(tmp_path, _synthetic_clip(frames=8, present=True))
    project_path = Path(settings.data_dir) / "project"
    _write_person_track(project_path, "track_repl", with_masks=with_masks)
    _write_character(project_path, "char_1")
    return settings


def test_person_track_masks_prefers_stored_segmentation(tmp_path):
    settings = _seed_replacement_project(tmp_path, with_masks=True)
    project_path = Path(settings.data_dir) / "project"
    masks = person_track_masks(project_path, "track_repl", 256, 256, 3)
    # Stored ellipse masks are non-rectangular; a box fallback would fill its bbox.
    mask = masks[0]
    nonzero = sum(mask.histogram()[1:])
    bbox = mask.getbbox()
    assert bbox is not None
    assert nonzero < (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * 0.95


def test_build_sidecar_marks_replacement_inactive_without_status(tmp_path):
    settings = _seed_replacement_project(tmp_path)
    request = video_request_from_job(_replacement_job(settings))
    asset = build_video_asset_sidecar(
        asset_id="asset_x",
        project_id="proj_1",
        generation_set_id="gen_x",
        request=request,
        job_id="job_replace_1",
        media_rel="assets/videos/out.mp4",
        created_at="2026-05-21T00:00:00Z",
        seed=1,
        target={"family": "ltx-video"},
        raw_settings={},
        adapter_id="ltx_pipelines",
        mime_type="video/mp4",
    )
    assert asset["recipe"]["normalizedSettings"]["replacementActive"] is False


def test_ltx_replacement_control_builds_segmentation_package(tmp_path):
    settings = _seed_replacement_project(tmp_path, with_masks=True)
    project_path = Path(settings.data_dir) / "project"
    adapter = LtxPipelinesVideoAdapter()
    adapter._settings = settings
    request = video_request_from_job(_replacement_job(settings))
    clip_path = project_path / "cache" / "control.webp"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    control = adapter._ltx_replacement_control(project_path, request, 9, clip_path)
    assert control.mask_mode == "segmentation"
    assert control.character_reference_count == 1
    assert control.person_tracking_active is True
    assert clip_path.exists()


def test_ltx_replace_person_active_run_marks_replacement_active(tmp_path, monkeypatch):
    settings = _seed_replacement_project(tmp_path, with_masks=True)
    project_path = Path(settings.data_dir) / "project"
    adapter = LtxPipelinesVideoAdapter()
    adapter._settings = settings
    adapter._resources_by_model["ltx_2_3"] = LtxPipelinesResources(
        checkpoint_path=Path("ckpt"),
        spatial_upsampler_path=Path("ups"),
        distilled_lora_path=Path("lora"),
        gemma_root=Path("gemma"),
    )
    monkeypatch.setattr(adapter, "_load_ltx_pipeline", lambda request, resources: object())
    # Real MP4 control-clip encoding (imageio/ffmpeg) is verified in-container; on
    # the host stub the writer so the test stays focused on the active-flag contract.
    monkeypatch.setattr(adapter, "_write_control_clip", lambda frames, path, fps: Path(path).write_bytes(b"clip"))

    def fake_encode(*, video, fps, audio, output_path, video_chunks_number):
        Path(output_path).write_bytes(b"fake-mp4")

    monkeypatch.setattr(
        adapter,
        "_run_ltx_pipeline",
        lambda **_kwargs: (None, None, 1, fake_encode),
    )

    job = _replacement_job(settings)
    request = video_request_from_job(job)
    result = adapter._run_real_ltx_video(
        settings=settings,
        job=job,
        request=request,
        progress=lambda *_a, **_k: None,
        cancel_requested=lambda: False,
    )
    normalized = result["assets"][0]["recipe"]["normalizedSettings"]
    assert normalized["replacementActive"] is True
    assert normalized["maskMode"] == "segmentation"
    assert normalized["replacementAdapter"] == "ltx_pipelines"
    assert normalized["personTrackId"] == "track_repl"


def test_ltx_replace_person_mock_run_stays_inactive(tmp_path, monkeypatch):
    settings = _seed_replacement_project(tmp_path, with_masks=True)
    adapter = LtxPipelinesVideoAdapter()
    adapter._settings = settings
    job = _replacement_job(settings, mock=True)
    request = video_request_from_job(job)
    result = adapter.run(
        settings=settings,
        job=job,
        request=request,
        progress=lambda *_a, **_k: None,
        cancel_requested=lambda: False,
    )
    asset = result["assets"][0]
    assert asset["recipe"]["normalizedSettings"]["replacementActive"] is False


def test_ensure_models_replace_person_requires_track_and_character(tmp_path):
    settings = _make_project(tmp_path, _synthetic_clip(frames=4, present=True))
    adapter = LtxPipelinesVideoAdapter()
    adapter._settings = settings
    request = video_request_from_job(_replacement_job(settings))
    with pytest.raises(RuntimeError, match="saved person track"):
        adapter.ensure_models(request)
