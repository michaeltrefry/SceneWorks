"""Unit coverage for the InstantID SDXL face-identity adapter (sc-2009).

Covers the pure engine logic + generate() guard rails without loading any real
models (no torch/diffusers/insightface). Full _run_pipeline / end-to-end identity
behavior is validated by the sc-2009 spike + a real worker job, not here.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from scene_worker.image_adapters import MODEL_TARGETS, create_image_adapter, image_request_from_job
from scene_worker.instantid_adapter import InstantIDAdapter, _letterbox

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
    assert target["guidanceScale"] == 5.0
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
