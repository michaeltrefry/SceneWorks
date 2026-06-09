"""Unit coverage for the InstantID SDXL face-identity adapter (sc-2009).

Covers the pure engine logic + generate() guard rails without loading any real
models (no torch/diffusers/insightface). Full _run_pipeline / end-to-end identity
behavior is validated by the sc-2009 spike + a real worker job, not here.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from PIL import Image

from scene_worker.image_adapters import MODEL_TARGETS, create_image_adapter, image_request_from_job
from scene_worker.instantid_adapter import (
    ANGLE_SET_ORDER,
    VIEW_ANGLE_KPS,
    InstantIDAdapter,
    _letterbox,
    _view_angle_kps,
)
from scene_worker.lora_adapters import LoraPipelineState
from scene_worker.openpose_skeleton import (
    draw_bodypose,
    draw_wholebody,
    face_box_from_keypoints,
    normalize_face,
    normalize_hands,
    normalize_keypoints,
)

_TEST_TARGET = {
    "label": "Test InstantID",
    "family": "sdxl",
    "supportsEdit": False,
    "steps": 30,
    "guidanceScale": 5.0,
    "variant": "fp16",
    "repo": "stabilityai/stable-diffusion-xl-base-1.0",
    "adapter": "instantid_sdxl",
    "instantId": {
        "repo": "InstantX/InstantID",
        "controlnetSubfolder": "ControlNetModel",
        "ipAdapter": "ip-adapter.bin",
    },
}

_NOOP = lambda *args, **kwargs: None  # noqa: E731


# The worker unit suite runs without torch installed; generate()/unload() do
# `importlib.import_module("torch")`. Patch it to this stand-in (with gpu_id="cpu",
# select_torch_device returns early and the cache-empty path is a guarded no-op).
class _FakeTorch:
    pass


def _patch_torch_import(monkeypatch):
    monkeypatch.setattr(
        "scene_worker.instantid_adapter.importlib.import_module",
        lambda name: _FakeTorch if name == "torch" else importlib.import_module(name),
    )


@pytest.fixture
def instantid_model(monkeypatch):
    monkeypatch.setitem(MODEL_TARGETS, "test_instantid", _TEST_TARGET)
    return "test_instantid"


def _job(model, **payload):
    base = {"projectId": "p", "mode": "character_image", "model": model, "prompt": "a person at a cafe"}
    base.update(payload)
    return {"id": "job_instantid", "payload": base}


def _generate(job):
    InstantIDAdapter().generate(
        settings=None,
        job=job,
        request=image_request_from_job(job),
        project_path=None,
        progress=_NOOP,
        cancel_requested=lambda: False,
    )


def test_letterbox_matches_target_and_preserves_aspect():
    # Square ref -> portrait output: padded, never stretched.
    out = _letterbox(Image.new("RGB", (1024, 1024)), 832, 1216)
    assert out.size == (832, 1216)
    # Wide ref -> square output.
    out2 = _letterbox(Image.new("RGB", (1600, 600)), 1024, 1024)
    assert out2.size == (1024, 1024)


def test_scale_defaults_and_overrides():
    adapter = InstantIDAdapter()
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={})) == 0.8
    assert adapter._controlnet_scale(SimpleNamespace(advanced={})) == 0.8
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 0.55})) == 0.55
    assert adapter._controlnet_scale(SimpleNamespace(advanced={"controlnetConditioningScale": 0.4})) == 0.4
    # Unparseable values fall back to the defaults.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": "x"})) == 0.8
    assert adapter._controlnet_scale(SimpleNamespace(advanced={"controlnetConditioningScale": None})) == 0.8


def test_steps_and_guidance_from_target():
    adapter = InstantIDAdapter()
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), _TEST_TARGET) == 30
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), _TEST_TARGET) == 80
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), _TEST_TARGET) == 5.0
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 3.5}), _TEST_TARGET) == 3.5


def test_rejects_non_instantid_model():
    with pytest.raises(RuntimeError, match="not an InstantID target"):
        _generate(_job("z_image_turbo"))


def test_rejects_edit_mode(instantid_model):
    with pytest.raises(RuntimeError, match="does not support image editing"):
        _generate(_job(instantid_model, mode="edit_image"))


def test_requires_reference_image(instantid_model):
    # character_image mode but no referenceAssetId -> guarded before any model load.
    with pytest.raises(RuntimeError, match="requires a character reference image"):
        _generate(_job(instantid_model))


def test_unload_when_empty_is_false():
    assert InstantIDAdapter().unload() is False


def test_builtin_realvisxl_target_is_wired():
    # The public sc-2009 target: model id matches the builtin manifest + web
    # constants, routes to the registered adapter, and carries the spike-validated
    # params. Guards the worker MODEL_TARGETS <-> adapter-registry wiring.
    target = MODEL_TARGETS["instantid_realvisxl"]
    assert target["adapter"] == InstantIDAdapter.id == "instantid_sdxl"
    assert target["family"] == "sdxl"
    assert target["supportsEdit"] is False
    assert target["repo"] == "SG161222/RealVisXL_V5.0"
    assert target["steps"] == 30
    assert target["guidanceScale"] == 3.0
    instant = target["instantId"]
    assert instant["repo"] == "InstantX/InstantID"
    assert instant["controlnetSubfolder"] == "ControlNetModel"
    assert instant["ipAdapter"] == "ip-adapter.bin"


def test_dispatch_routes_instantid_model_to_adapter():
    # Regression: the model -> adapter dispatch must route instantid_realvisxl to
    # the InstantID adapter, NOT fall through to the procedural placeholder.
    job = {"id": "j", "payload": {"model": "instantid_realvisxl", "mode": "character_image"}}
    # No adapters dict (test/direct path) -> lazily constructed.
    assert isinstance(create_image_adapter(job, None), InstantIDAdapter)
    # Runtime path: resolves the registered instance from the adapters dict.
    registered = InstantIDAdapter()
    assert create_image_adapter(job, {"instantid_sdxl": registered}) is registered


def test_dispatch_accepts_instantid_adapter_override():
    # SCENEWORKS_IMAGE_ADAPTER=instantid_sdxl must be an accepted explicit override.
    job = {"id": "j", "payload": {"model": "z_image_turbo", "adapter": "instantid_sdxl"}}
    assert isinstance(create_image_adapter(job, None), InstantIDAdapter)


def test_missing_extras_raises_actionable_error(instantid_model, monkeypatch):
    # When the optional extras are absent, generate() must fail with an install hint
    # (not a raw ModuleNotFoundError mid-run). Simulate insightface being missing.
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: None if name == "insightface" else object()
    )
    with pytest.raises(RuntimeError, match="requirements-instantid.txt"):
        _generate(_job(instantid_model, referenceAssetId="ref-x"))


def test_view_angle_pack_is_well_formed():
    # The canonical pack covers the 11 shipped angles, all normalized to [0,1].
    expected = {
        "front", "three_quarter_left", "three_quarter_right", "left_profile", "right_profile",
        "up", "down", "up_left", "up_right", "down_left", "down_right",
    }
    assert expected <= set(VIEW_ANGLE_KPS)
    for pts in VIEW_ANGLE_KPS.values():
        assert len(pts) == 5
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in pts)
    scaled = _view_angle_kps("front", 1024)
    assert scaled.shape == (5, 2) and float(scaled.max()) <= 1024.0
    assert _view_angle_kps("nonexistent", 1024) is None


def test_view_angle_resolves_only_known_angles():
    assert InstantIDAdapter._view_angle(SimpleNamespace(advanced={"viewAngle": "left_profile"})) == "left_profile"
    assert InstantIDAdapter._view_angle(SimpleNamespace(advanced={"viewAngle": "bogus"})) is None
    assert InstantIDAdapter._view_angle(SimpleNamespace(advanced={})) is None


def test_angle_set_order_covers_pack_without_dupes():
    # The one-click angle set generates every packed angle exactly once, front first.
    assert len(ANGLE_SET_ORDER) == 11
    assert len(set(ANGLE_SET_ORDER)) == len(ANGLE_SET_ORDER)
    assert ANGLE_SET_ORDER[0] == "front"
    assert all(angle in VIEW_ANGLE_KPS for angle in ANGLE_SET_ORDER)


# ---- pose library (sc-2064 / sc-2065) -------------------------------------------

# A minimal front-standing COCO-18 skeleton (normalized) for the renderer/face-box tests.
_FRONT_KPS = [
    [0.50, 0.09], [0.50, 0.16], [0.44, 0.17], [0.42, 0.29], [0.41, 0.40],
    [0.56, 0.17], [0.58, 0.29], [0.59, 0.40], [0.47, 0.47], [0.46, 0.70],
    [0.46, 0.93], [0.53, 0.47], [0.54, 0.70], [0.54, 0.93], [0.48, 0.078],
    [0.52, 0.078], [0.46, 0.088], [0.54, 0.088],
]


def test_normalize_keypoints_coerces_to_18():
    # [x,y], [x,y,conf], None, and conf<=0 are all handled; result is always length 18.
    raw = [[0.5, 0.1], [0.4, 0.2, 1.0], None, [0.3, 0.3, 0.0]]
    out = normalize_keypoints(raw)
    assert len(out) == 18
    assert out[0] == (0.5, 0.1)
    assert out[1] == (0.4, 0.2)
    assert out[2] is None  # explicit None
    assert out[3] is None  # confidence 0 -> dropped
    assert all(p is None for p in out[4:])  # padded to 18
    assert normalize_keypoints(None) == [None] * 18


def test_draw_bodypose_renders_colored():
    # Renders an OpenPose control image of the requested size with colored joints/limbs.
    skel = draw_bodypose(64, 96, normalize_keypoints(_FRONT_KPS))
    assert skel.shape == (96, 64, 3)
    assert str(skel.dtype) == "uint8"
    assert skel.any()  # not an all-black canvas
    # An all-empty skeleton stays black.
    assert not draw_bodypose(64, 96, [None] * 18).any()


def test_normalize_hands_and_face():
    # A [left, right] pair is preserved as two 21-point hands.
    pair = normalize_hands([[[0.1, 0.1]] * 21, [[0.2, 0.2]] * 21])
    assert pair is not None and len(pair) == 2 and len(pair[0]) == 21 and len(pair[1]) == 21
    # A flat 42-point list splits into two 21-point hands.
    flat = normalize_hands([[0.3, 0.3]] * 42)
    assert flat is not None and len(flat) == 2 and len(flat[0]) == 21 and len(flat[1]) == 21
    # Empty / None / non-list -> None.
    assert normalize_hands(None) is None and normalize_hands([]) is None
    # Face coerces to exactly 68 points (extra trimmed); None -> None.
    assert len(normalize_face([[0.5, 0.4]] * 70)) == 68
    assert normalize_face(None) is None and normalize_face([]) is None


def test_draw_wholebody_adds_hands_and_face():
    body = normalize_keypoints(_FRONT_KPS)
    # Body-only path is byte-identical to draw_bodypose (InstantID/Qwen unaffected).
    body_only = draw_wholebody(128, 192, body, stickwidth=4)
    assert (body_only == draw_bodypose(128, 192, body, stickwidth=4)).all()
    # Hands (off to the sides) + a face cluster (above the head) paint extra marks,
    # including pure-white face dots that the body/hand palette never produces.
    hands = [[[0.05, 0.5]] * 21, [[0.95, 0.5]] * 21]
    face = [[0.5, 0.03]] * 68
    whole = draw_wholebody(128, 192, body, hands=hands, face=face, stickwidth=4)
    assert int((whole.sum(2) > 0).sum()) > int((body_only.sum(2) > 0).sum())
    assert ((whole[:, :, 0] == 255) & (whole[:, :, 1] == 255) & (whole[:, :, 2] == 255)).any()


def test_face_box_from_keypoints():
    # A head (nose/eyes/neck present) yields a (cx, cy, height_frac) box; no head -> None
    # so the adapter disables IdentityNet + the face-restoration pass.
    box = face_box_from_keypoints(normalize_keypoints(_FRONT_KPS))
    assert box is not None and len(box) == 3
    cx, cy, fhf = box
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert 0.045 <= fhf <= 0.20  # clamped small full-body face fraction
    # Head keypoints (nose=0, eyes=14/15) absent -> None.
    headless = list(normalize_keypoints(_FRONT_KPS))
    for i in (0, 14, 15):
        headless[i] = None
    assert face_box_from_keypoints(headless) is None


def test_realvisxl_target_has_openpose_for_pose_library():
    open_pose = MODEL_TARGETS["instantid_realvisxl"].get("openPose")
    assert open_pose and open_pose["repo"] == "xinsir/controlnet-openpose-sdxl-1.0"


def test_openpose_scale_default_and_override():
    adapter = InstantIDAdapter()
    assert adapter._openpose_scale(SimpleNamespace(advanced={})) == 0.7
    assert adapter._openpose_scale(SimpleNamespace(advanced={"openPoseScale": 0.55})) == 0.55
    assert adapter._openpose_scale(SimpleNamespace(advanced={"openPoseScale": "x"})) == 0.7


def test_face_restore_enabled_default_and_toggle():
    # Defaults off; explicit booleans + string forms toggle the full-body restoration pass.
    assert InstantIDAdapter._face_restore_enabled(SimpleNamespace(advanced={})) is False
    assert InstantIDAdapter._face_restore_enabled(SimpleNamespace(advanced={"faceRestore": False})) is False
    assert InstantIDAdapter._face_restore_enabled(SimpleNamespace(advanced={"faceRestore": True})) is True
    assert InstantIDAdapter._face_restore_enabled(SimpleNamespace(advanced={"faceRestore": "false"})) is False
    assert InstantIDAdapter._face_restore_enabled(SimpleNamespace(advanced={"faceRestore": "off"})) is False


# ---- SDXL LoRA application (sc-2224) --------------------------------------------
#
# The InstantID pipe is a StableDiffusionXLControlNetPipeline; the sc-2222 spike
# confirmed the existing SDXL LoRA merge path stacks on it (IdentityNet/OpenPose +
# IP-Adapter, bf16/MPS) and persists across the per-pose _restore_face pass. These
# tests cover the adapter-level wiring (apply once before the angle/pose loop, state
# threaded across jobs, reset on unload) with the pipe + merge mocked — the merge
# math + family gating live in lora_adapters and are exercised via the real
# apply_loras_to_pipeline in test_lora_family_incompatibility_is_rejected.


def _generate_capturing_loras(monkeypatch, adapter, job):
    """Run generate() with the model load + LoRA merge + asset writer mocked, returning
    the list of recorded apply_loras_to_pipeline calls."""
    calls: list[dict] = []
    applied_state = LoraPipelineState(key="applied", adapter_names=("sw_test",))

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())  # extras present
    _patch_torch_import(monkeypatch)
    monkeypatch.setattr(adapter, "_load_pipeline", lambda *a, **k: SimpleNamespace(tag="fakepipe"))
    monkeypatch.setattr("scene_worker.instantid_adapter.select_torch_device", lambda *a, **k: "cpu")
    monkeypatch.setattr("scene_worker.instantid_adapter.activate_torch_device", lambda *a, **k: None)

    def _fake_apply(pipe, loras, **kwargs):
        calls.append({"pipe": pipe, "loras": loras, **kwargs})
        return applied_state

    monkeypatch.setattr("scene_worker.instantid_adapter.apply_loras_to_pipeline", _fake_apply)
    monkeypatch.setattr(
        "scene_worker.instantid_adapter.ImageAssetWriter.write_incremental_outputs",
        lambda self, **kwargs: {"images": []},
    )
    adapter.generate(
        settings=SimpleNamespace(gpu_id="cpu"),
        job=job,
        request=image_request_from_job(job),
        project_path=None,
        progress=_NOOP,
        cancel_requested=lambda: False,
    )
    return calls, applied_state


def test_generate_applies_request_loras(instantid_model, monkeypatch):
    loras = [{"id": "kelsie-sdxl", "path": "/loras/kelsie.safetensors", "families": ["sdxl"]}]
    job = _job(instantid_model, referenceAssetId="ref-x", loras=loras)
    adapter = InstantIDAdapter()
    calls, applied_state = _generate_capturing_loras(monkeypatch, adapter, job)

    assert len(calls) == 1
    call = calls[0]
    assert call["loras"] == loras
    assert call["adapter_id"] == "instantid_sdxl"
    assert call["model_family"] == "sdxl"
    assert call["model_id"] == instantid_model
    # Fresh adapter: the first job threads in the empty starting state.
    assert call["previous_state"] == LoraPipelineState()
    # The returned state is stored so the next job can clear/diff against it.
    assert adapter._loaded_lora_state is applied_state


def test_generate_with_no_loras_still_applies_empty(instantid_model, monkeypatch):
    # A job without loras must still call apply (with []), so a LoRA from a prior job on
    # the cached pipe is cleared rather than silently carried over.
    job = _job(instantid_model, referenceAssetId="ref-x")
    calls, _ = _generate_capturing_loras(monkeypatch, InstantIDAdapter(), job)
    assert len(calls) == 1 and calls[0]["loras"] == []


def test_generate_threads_previous_lora_state_across_jobs(instantid_model, monkeypatch):
    # The cached pipe persists across jobs, so each apply must receive the prior job's
    # returned state as previous_state (the diff drives load/clear of adapters).
    adapter = InstantIDAdapter()
    job = _job(instantid_model, referenceAssetId="ref-x", loras=[{"id": "a", "path": "/a.safetensors"}])
    calls1, applied_state = _generate_capturing_loras(monkeypatch, adapter, job)
    assert calls1[0]["previous_state"] == LoraPipelineState()

    calls2, _ = _generate_capturing_loras(monkeypatch, adapter, job)
    assert calls2[0]["previous_state"] is applied_state


def test_unload_resets_lora_state(monkeypatch):
    _patch_torch_import(monkeypatch)  # unload() evaluates importlib.import_module("torch")
    adapter = InstantIDAdapter()
    adapter._pipe = SimpleNamespace(tag="fakepipe")
    adapter._loaded_lora_state = LoraPipelineState(key="applied", adapter_names=("sw_test",))
    adapter._empty_cache = lambda *_a, **_k: None  # the cache-empty call itself is a no-op
    assert adapter.unload() is True
    # The merge belongs to the (now-discarded) pipe; the next pipe must start clean.
    assert adapter._loaded_lora_state == LoraPipelineState()


def test_lora_family_incompatibility_is_rejected():
    # The InstantID family is "sdxl"; a flux LoRA must be rejected. validate runs before
    # the pipe is touched, so a bare sentinel pipe is never used.
    from scene_worker.lora_adapters import apply_loras_to_pipeline

    flux_lora = [{"id": "flux-style", "path": "/loras/flux.safetensors", "families": ["flux"]}]
    with pytest.raises(RuntimeError, match="not compatible"):
        apply_loras_to_pipeline(
            object(), flux_lora, adapter_id="instantid_sdxl", model_family="sdxl"
        )
