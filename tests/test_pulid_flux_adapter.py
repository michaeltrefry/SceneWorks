"""Unit coverage for the PuLID-FLUX face-identity adapter (sc-2012, epic 2003).

Covers the pure engine logic + generate() guard rails + adapter dispatch without
loading any real models (no torch / flux / pulid / insightface). Real
identity behavior is validated by the sc-2012 hardware spike + a worker job,
not here.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from scene_worker.image_adapters import MODEL_TARGETS, create_image_adapter, image_request_from_job
from scene_worker.pulid_flux_adapter import PuLIDFluxAdapter

_TEST_TARGET = {
    "label": "Test PuLID-FLUX",
    "family": "flux",
    "supportsEdit": False,
    "steps": 30,
    "guidanceScale": 4.0,
    "maxSequenceLength": 128,
    "repo": "black-forest-labs/FLUX.1-dev",
    "adapter": "pulid_flux",
    "pulidFlux": {
        "repo": "guozinan/PuLID",
        "weight": "pulid_flux_v0.9.1.safetensors",
        "version": "v0.9.1",
        "bflConfig": "flux-dev",
        "maxSequenceLength": 128,
    },
}

_NOOP = lambda *args, **kwargs: None  # noqa: E731


@pytest.fixture
def pulid_flux_model(monkeypatch):
    monkeypatch.setitem(MODEL_TARGETS, "test_pulid_flux", _TEST_TARGET)
    return "test_pulid_flux"


def _job(model, **payload):
    base = {"projectId": "p", "mode": "character_image", "model": model, "prompt": "a person at a cafe"}
    base.update(payload)
    return {"id": "job_pulid_flux", "payload": base}


def _generate(job):
    PuLIDFluxAdapter().generate(
        settings=None,
        job=job,
        request=image_request_from_job(job),
        project_path=None,
        progress=_NOOP,
        cancel_requested=lambda: False,
    )


def test_id_weight_default_and_overrides():
    adapter = PuLIDFluxAdapter()
    # The PuLID "photoreal" default — sc-2012 measured 0.8016 ArcFace cosine at this setting.
    assert adapter._id_weight(SimpleNamespace(advanced={})) == 1.0
    assert adapter._id_weight(SimpleNamespace(advanced={"idWeight": 0.7})) == 0.7
    # Clamped to the upstream slider band [0.0, 3.0].
    assert adapter._id_weight(SimpleNamespace(advanced={"idWeight": -0.5})) == 0.0
    assert adapter._id_weight(SimpleNamespace(advanced={"idWeight": 9.0})) == 3.0
    # Unparseable values fall back to the default.
    assert adapter._id_weight(SimpleNamespace(advanced={"idWeight": "x"})) == 1.0
    assert adapter._id_weight(SimpleNamespace(advanced={"idWeight": None})) == 1.0


def test_timestep_to_start_cfg_default_and_clamp():
    adapter = PuLIDFluxAdapter()
    # Upstream "photoreal" recommendation; matches the sc-2012 best-run setting.
    assert adapter._timestep_to_start_cfg(SimpleNamespace(advanced={})) == 4
    assert adapter._timestep_to_start_cfg(SimpleNamespace(advanced={"timestepToStartCfg": 0})) == 0
    assert adapter._timestep_to_start_cfg(SimpleNamespace(advanced={"timestepToStartCfg": 6})) == 6
    # safe_int clamps to [0, 20].
    assert adapter._timestep_to_start_cfg(SimpleNamespace(advanced={"timestepToStartCfg": -3})) == 0
    assert adapter._timestep_to_start_cfg(SimpleNamespace(advanced={"timestepToStartCfg": 999})) == 20


def test_steps_and_guidance_from_target():
    adapter = PuLIDFluxAdapter()
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), _TEST_TARGET) == 30
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), _TEST_TARGET) == 80
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), _TEST_TARGET) == 4.0
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 3.0}), _TEST_TARGET) == 3.0


def test_rejects_non_pulid_flux_model():
    with pytest.raises(RuntimeError, match="not a PuLID-FLUX target"):
        _generate(_job("z_image_turbo"))


def test_rejects_edit_mode(pulid_flux_model):
    with pytest.raises(RuntimeError, match="does not support image editing"):
        _generate(_job(pulid_flux_model, mode="edit_image"))


def test_requires_reference_image(pulid_flux_model):
    # character_image mode but no referenceAssetId -> guarded before any model load.
    with pytest.raises(RuntimeError, match="requires a character reference image"):
        _generate(_job(pulid_flux_model))


def test_unload_when_empty_is_false():
    assert PuLIDFluxAdapter().unload() is False


def test_unload_routes_through_gc_ordered_release(monkeypatch):
    # sc-4192: a non-empty unload must free the ~37 GB stack via
    # release_inference_memory (gc.collect() BEFORE empty_cache()), not a bare
    # empty_cache() that leaves reference-cycled weights resident.
    adapter = PuLIDFluxAdapter()
    adapter._flow_model = SimpleNamespace(tag="fake")
    fake_torch = object()
    monkeypatch.setattr(
        "scene_worker.pulid_flux_adapter.importlib.import_module",
        lambda name: fake_torch if name == "torch" else importlib.import_module(name),
    )
    calls = []
    monkeypatch.setattr(
        "scene_worker.pulid_flux_adapter.release_inference_memory",
        lambda torch: calls.append(torch),
    )
    assert adapter.unload() is True
    assert calls == [fake_torch]
    assert adapter._flow_model is None


def test_builtin_pulid_flux_target_is_wired():
    # The public sc-2012 target: model id matches the builtin manifest + web
    # constants, routes to the registered adapter, and carries the spike-validated
    # params. Guards the worker MODEL_TARGETS <-> adapter-registry wiring.
    target = MODEL_TARGETS["pulid_flux_dev"]
    assert target["adapter"] == PuLIDFluxAdapter.id == "pulid_flux"
    assert target["family"] == "flux"
    assert target["supportsEdit"] is False
    assert target["repo"] == "black-forest-labs/FLUX.1-dev"
    # sc-2012 spike defaults (the PuLID "photoreal" preset).
    assert target["steps"] == 30
    assert target["guidanceScale"] == 4.0
    assert target["maxSequenceLength"] == 128
    pulid_cfg = target["pulidFlux"]
    assert pulid_cfg["repo"] == "guozinan/PuLID"
    assert pulid_cfg["weight"] == "pulid_flux_v0.9.1.safetensors"
    assert pulid_cfg["version"] == "v0.9.1"
    assert pulid_cfg["bflConfig"] == "flux-dev"


def test_dispatch_routes_pulid_flux_model_to_adapter():
    # Regression: the model -> adapter dispatch must route pulid_flux_dev to the
    # PuLID-FLUX adapter, NOT fall through to the procedural placeholder or to
    # the regular FluxDiffusersAdapter (which would silently try to run plain
    # FLUX without identity conditioning).
    job = {"id": "j", "payload": {"model": "pulid_flux_dev", "mode": "character_image"}}
    # No adapters dict (test/direct path) -> lazily constructed.
    assert isinstance(create_image_adapter(job, None), PuLIDFluxAdapter)
    # Runtime path: resolves the registered instance from the adapters dict.
    registered = PuLIDFluxAdapter()
    assert create_image_adapter(job, {"pulid_flux": registered}) is registered


def test_dispatch_accepts_pulid_flux_adapter_override():
    # SCENEWORKS_IMAGE_ADAPTER=pulid_flux must be an accepted explicit override.
    job = {"id": "j", "payload": {"model": "z_image_turbo", "adapter": "pulid_flux"}}
    assert isinstance(create_image_adapter(job, None), PuLIDFluxAdapter)


def test_missing_extras_raises_actionable_error(pulid_flux_model, monkeypatch):
    # When the optional extras are absent, generate() must fail with an install hint
    # (not a raw ModuleNotFoundError mid-run). Simulate facexlib being missing —
    # it's PuLID-specific (not shared with InstantID), so it's the canonical signal.
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: None if name == "facexlib" else object()
    )
    with pytest.raises(RuntimeError, match="requirements-pulid-flux.txt"):
        _generate(_job(pulid_flux_model, referenceAssetId="ref-x"))


def test_builtin_pulid_flux_target_caps_memory_floor():
    # sc-2012 spike measured ~85 GB peak unified memory on MPS bf16 at 1024². A
    # 64 GB Mac is feasible but tight; the manifest's minMemoryGb gates the model
    # out of the picker on hosts below the bar.
    # (The MODEL_TARGETS dict carries the resolver knobs; the manifest in
    # config/manifests/builtin.models.jsonc carries the UI/limits gating — both
    # need to agree, validated by the model snapshot test in parity.)
    target = MODEL_TARGETS["pulid_flux_dev"]
    assert target["adapter"] == "pulid_flux"
