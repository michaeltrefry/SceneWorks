"""Unit tests for the DWPose detector adapter (sc-2285).

Cover the pure COCO-WholeBody-133 -> SceneWorks OpenPose conversion, backend
availability gating, and the orchestration (run_pose_detect) with a fake detector
so coverage needs neither the onnx weights nor a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scene_worker import pose_adapters as pa


def _fake_wholebody(n: int = 1, *, score: float = 5.0):
    """(keypoints, scores) like rtmlib Wholebody: (N,133,2) px + (N,133)."""
    kps = np.zeros((n, 133, 2), dtype=np.float32)
    # place every keypoint at a distinct, in-frame location so conversion is checkable
    for i in range(133):
        kps[:, i] = (100 + i, 200 + i)
    scores = np.full((n, 133), score, dtype=np.float32)
    return kps, scores


def test_wholebody_to_openpose_shapes_and_neck():
    kps, sc = _fake_wholebody(1)
    rec = pa.wholebody_to_openpose(kps[0], sc[0], w=1000, h=1000)

    assert len(rec["keypoints"]) == 18
    assert [len(h) for h in rec["hands"]] == [21, 21]
    assert len(rec["face"]) == 68

    # nose (openpose 0) == coco 0 ; normalized to [0,1]
    assert rec["keypoints"][0][0] == pytest.approx(100 / 1000)
    # neck (openpose 1) == midpoint of shoulders (coco 5 & 6)
    exp_x = ((100 + 5) + (100 + 6)) / 2 / 1000
    assert rec["keypoints"][1][0] == pytest.approx(exp_x)
    # right wrist (openpose 4) maps from coco 10
    assert rec["keypoints"][4][0] == pytest.approx((100 + 10) / 1000)
    # confidence carried through
    assert rec["keypoints"][0][2] == pytest.approx(5.0)


def test_square_fit_centers_into_largest_square():
    from scene_worker.openpose_skeleton import square_fit

    assert square_fit(100, 100) == (100, 0, 0)  # square: unchanged (no regression)
    assert square_fit(200, 100) == (100, 50, 0)  # wide: centered horizontally
    assert square_fit(100, 200) == (100, 0, 50)  # tall: centered vertically


def test_squareify_pads_short_axis_preserving_proportions():
    rec = {
        "keypoints": [[0.5, 0.5, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        "hands": [[None], [None]],
        "face": [[0.5, 0.5, 0.9]],
    }
    # Portrait 320x480 -> square side 480, x padded by (480-320)/2 = 80, y unchanged.
    out = pa.squareify(rec, 320, 480)
    assert out["keypoints"][0] == pytest.approx([0.5, 0.5, 1.0])  # center stays center
    assert out["keypoints"][1][0] == pytest.approx(80 / 480)  # left edge moves toward center
    assert out["keypoints"][1][1] == pytest.approx(0.0)  # y keeps full range
    assert out["keypoints"][2][0] == pytest.approx((320 + 80) / 480)
    assert out["keypoints"][2][1] == pytest.approx(1.0)
    # The x-span equals the source width fraction (320/480) — proportions preserved,
    # NOT stretched to fill the square.
    assert out["keypoints"][2][0] - out["keypoints"][1][0] == pytest.approx(320 / 480)
    assert out["face"][0] == pytest.approx([0.5, 0.5, 0.9])
    assert out["hands"][0][0] is None  # None carried through


def test_pose_detector_backend_available(monkeypatch):
    monkeypatch.setattr(pa, "_module_available", lambda name: True)
    assert pa.pose_detector_backend_available() is True
    monkeypatch.setattr(pa, "_module_available", lambda name: name != "rtmlib")
    assert pa.pose_detector_backend_available() is False


def test_require_pose_extras_raises_clearly(monkeypatch):
    monkeypatch.setattr(pa, "_module_available", lambda name: False)
    with pytest.raises(pa.PoseDetectError, match="rtmlib"):
        pa._require_pose_extras()


def test_run_pose_detect_no_sources_raises():
    with pytest.raises(pa.PoseDetectError, match="No source images"):
        pa.run_pose_detect(SimpleNamespace(data_dir=".", gpu_id="cpu"), {"id": "j", "payload": {}})


def test_resolve_source_path_absolute_asset_and_relative(tmp_path, monkeypatch):
    # An absolute, existing path is used as-is (spike/tests).
    direct = tmp_path / "a.png"
    direct.write_bytes(b"x")
    assert pa._resolve_source_path({"path": str(direct)}, None) == str(direct)

    # An assetId resolves against the project via the shared loader (Create tab).
    media = tmp_path / "assets" / "images" / "img.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"y")
    import sceneworks_shared

    monkeypatch.setattr(
        sceneworks_shared,
        "load_asset_with_media",
        lambda project_path, asset_id: ({"id": asset_id}, media),
        raising=False,
    )
    assert pa._resolve_source_path({"assetId": "asset_9"}, tmp_path) == str(media)

    # A project-relative path falls back to a project-root join.
    assert pa._resolve_source_path({"path": "assets/images/img.png"}, tmp_path) == str(media)

    # Nothing resolvable -> None so the source is reported unreadable, not a crash.
    assert pa._resolve_source_path({"assetId": "missing"}, None) is None


def test_run_pose_detect_with_fake_detector(tmp_path):
    cv2 = pytest.importorskip("cv2")
    # a real (blank) source image on disk so cv2.imread succeeds
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), np.full((480, 320, 3), 30, dtype=np.uint8))

    runtime = pa.PoseDetectorRuntime(
        model=lambda img: _fake_wholebody(1), device="cpu", detector_id="fake/test"
    )
    settings = SimpleNamespace(data_dir=str(tmp_path), gpu_id="cpu")
    job = {
        "id": "job_pose_test",
        "payload": {"sources": [{"path": str(src), "assetId": "asset_1"}], "minConf": 0.3},
    }
    events: list = []
    result = pa.run_pose_detect(
        settings, job,
        progress=lambda *a: events.append(a),
        detector_factory=lambda s: runtime,
    )

    assert result["poseDetectionActive"] is True
    assert result["detector"]["device"] == "cpu"
    assert len(result["sources"]) == 1
    source = result["sources"][0]
    assert source["sourceWidth"] == 320 and source["sourceHeight"] == 480
    assert source["sourceAspect"] == pytest.approx(320 / 480, abs=1e-3)
    assert source["sourceAssetId"] == "asset_1"
    assert len(source["poses"]) == 1
    pose = source["poses"][0]
    assert len(pose["keypoints"]) == 18
    assert [len(h) for h in pose["hands"]] == [21, 21]
    assert len(pose["face"]) == 68
    assert pose["facing"] in {"front", "back", "profile"}
    # skeleton preview rendered to the job-scoped staging dir
    from pathlib import Path
    assert Path(pose["skeletonPreview"]).exists()
    # Preview is rendered SQUARE — poses are stored square-canonical (epic 2282).
    prev = cv2.imread(pose["skeletonPreview"])
    assert prev is not None and prev.shape[0] == prev.shape[1]
    assert events  # progress was reported


def test_run_pose_detect_deletes_temp_uploads(tmp_path):
    cv2 = pytest.importorskip("cv2")
    # A File-Upload source staged under <data_dir>/cache/pose-uploads (transient).
    uploads = tmp_path / "cache" / "pose-uploads"
    uploads.mkdir(parents=True)
    temp_src = uploads / "upload-abc.png"
    cv2.imwrite(str(temp_src), np.full((480, 320, 3), 30, dtype=np.uint8))
    # A non-temp source elsewhere must be left untouched.
    keep = tmp_path / "keep.png"
    cv2.imwrite(str(keep), np.full((480, 320, 3), 30, dtype=np.uint8))

    runtime = pa.PoseDetectorRuntime(
        model=lambda img: _fake_wholebody(1), device="cpu", detector_id="fake/test"
    )
    settings = SimpleNamespace(data_dir=str(tmp_path), gpu_id="cpu")
    job = {
        "id": "job_pose_temp",
        "payload": {"sources": [{"path": str(temp_src), "temp": True}, {"path": str(keep)}]},
    }
    pa.run_pose_detect(settings, job, detector_factory=lambda s: runtime)

    assert not temp_src.exists()  # temp upload removed after detection
    assert keep.exists()  # non-temp source untouched
