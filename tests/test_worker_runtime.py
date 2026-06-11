from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest

from scene_worker.adapter_utils import cancel_step_callback, filter_call_kwargs
from scene_worker.hf_cache import safe_repo_dir_name
from scene_worker.caption_adapters import (
    JOY_CAPTION_RESAMPLE,
    JoyCaptionOptions,
    build_joy_caption_prompt,
    caption_with_trigger_words,
    normalize_processor_resample,
)
from scene_worker.image_adapters import (
    AuraSrUpscaler,
    ChromaDiffusersAdapter,
    CHARACTER_ANGLE_SET_ORDER,
    FluxDiffusersAdapter,
    ImageAssetWriter,
    KolorsDiffusersAdapter,
    LensTurboAdapter,
    MODEL_TARGETS,
    QwenImageAdapter,
    REAL_ESRGAN_MODEL_SPECS,
    RealEsrganUpscaler,
    SdxlDiffusersAdapter,
    ZImageDiffusersAdapter,
    create_image_adapter,
    create_image_upscaler,
    emit_worker_event,
    fit_image,
    format_batch_running_message,
    gpu_memory_snapshot,
    huggingface_repo_cache_path,
    image_batch_progress,
    image_request_from_job,
    lens_resolution_for,
    load_mask_image,
    load_reference_image,
    load_source_image,
    model_supports_detail,
    model_supports_edit,
    model_supports_inpaint,
    normalize_fit_mode,
    outpaint_border_mask,
    pipeline_component_devices,
    require_inference_backend_for_gpu_worker,
    interleave_resolution_for,
    resolve_seed,
    select_torch_device,
    sensenova_resolution_for,
    SenseNovaU1Adapter,
    verify_pipeline_on_device,
)
from scene_worker.upscalers import (
    AuraSRUpscaler,
    RealESRGANUpscaler,
    TileSlice,
    UpscaleJob,
    _load_state_dict,
    create_upscaler_engine,
    tile_slices,
)
from scene_worker.lora_adapters import (
    LoraSpec,
    adapter_network_type,
    apply_loras_to_pipeline,
    clear_loras,
    first_safetensors_path,
    lora_cache_key,
    lora_weight,
    normalize_lora_specs,
    reject_loras_if_unsupported,
    reject_lokr_loras,
    resolve_lora_file,
    set_adapter_weights_on_module,
    validate_lora_compatibility,
)
from scene_worker.runtime import (
    FORCED_CANCEL_EXIT_CODE,
    JobCancelMonitor,
    child_environment,
    friendly_failure,
    heartbeat,
    is_cuda_oom,
    keep_job_alive,
    loaded_models_from_adapters,
    main,
    resolve_loaded_models,
    run_check,
    run_lora_train_job,
    run_prompt_refine_job,
    run_video_job,
    worker_capabilities,
)
from scene_worker.prompt_refine import (
    PromptRefineUnavailable,
    PromptRefiner,
    build_system_prompt,
    clean_output,
)
from scene_worker.training_adapters import (
    SUPPORTED_LR_SCHEDULERS,
    SUPPORTED_TRAINING_PLAN_VERSION,
    LensLoraTrainer,
    SdxlLoraTrainer,
    KolorsLoraTrainer,
    TrainingKernelError,
    WanLoraTrainer,
    WanMoeLoraTrainer,
    ZImageLoraTrainer,
    _KolorsLoraBackend,
    _SdxlLoraBackend,
    _WanLoraBackend,
    _WanMoeLoraBackend,
    _ZImageLoraBackend,
    apply_weight_noise,
    build_lr_scheduler,
    build_optimizer,
    build_peft_network_config,
    bucket_resolution,
    create_training_kernel,
    dry_run_training_summary,
    flow_matching_velocity_target,
    lr_decay_multiplier,
    lr_schedule_updates,
    normalize_lr_scheduler,
    read_run_config,
    resolve_pretrained_source,
    resolve_training_adapter_source,
    sample_training_timestep,
    seeded_sample,
    training_adapter_weight_name,
    validate_training_plan,
    write_lokr_adapter,
)
from scene_worker.video_adapters import (
    DiffusersVideoAdapter,
    LtxPipelinesVideoAdapter,
    VIDEO_MODEL_TARGETS,
    VendorPatchDriftError,
    _require_patch_target,
    character_reference_images,
    create_video_adapter,
    evenly_spaced_indices,
    frames_from_output,
    install_ltx_pipelines_multigpu_compat,
    ltx_frame_count,
    ltx_mps_gating,
    load_seekable_image_frame,
    person_track_masks,
    safe_download_dir,
    video_generation_result,
    video_request_from_job,
)


class AcceptsNone:
    def __call__(self, *, prompt, image=None):
        return prompt, image


class FakeLoraPipe:
    def __init__(self):
        self.loaded = []
        self.set_calls = []
        self.unloaded = 0

    def load_lora_weights(self, path, adapter_name=None):
        self.loaded.append((path, adapter_name))

    def set_adapters(self, names, adapter_weights=None):
        self.set_calls.append((names, adapter_weights))

    def unload_lora_weights(self):
        self.unloaded += 1


class FakeTargetedLoraPipe(FakeLoraPipe):
    def __init__(self):
        super().__init__()
        self.deleted = []

    def delete_adapters(self, names):
        self.deleted.append(names)


class FakeDenoiserModule:
    """Stand-in for a pipeline's unet/transformer that records module-level
    set_adapters calls (the LoKr weight path)."""

    def __init__(self):
        self.set_calls = []

    def set_adapters(self, names, weights=None):
        self.set_calls.append((list(names), list(weights) if weights is not None else None))


class FakeLokrPipe(FakeTargetedLoraPipe):
    """A pipe exposing a denoiser module so LoKr can inject into it."""

    def __init__(self):
        super().__init__()
        self.unet = FakeDenoiserModule()


class FakeMoeLoraPipe(FakeLoraPipe):
    """Two-expert (A14B) pipe: has a transformer_2 and records whether each
    load_lora_weights call targeted it (load_into_transformer_2=True)."""

    transformer_2 = object()

    def load_lora_weights(self, path, adapter_name=None, **kwargs):
        self.loaded.append((path, adapter_name, bool(kwargs.get("load_into_transformer_2", False))))


class FakeSingleLoraPipe:
    def __init__(self):
        self.loaded = []

    def load_lora_weights(self, path, adapter_name=None):
        self.loaded.append((path, adapter_name))


class FakePeftBackendErrorPipe:
    def load_lora_weights(self, path, adapter_name=None):
        raise ValueError("PEFT backend is required for this method.")


def test_cpu_worker_does_not_advertise_gpu_generation_capabilities():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert capabilities == ["cpu"]
    assert "placeholder" not in capabilities


def test_gpu_worker_advertises_generation_capabilities(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})

    assert "image_generate" in capabilities
    assert "image_vqa" in capabilities
    assert "video_generate" in capabilities
    assert "training_caption" in capabilities
    assert "person_replace" in capabilities
    assert "placeholder" not in capabilities


def test_gpu_worker_without_cuda_torch_does_not_claim_generation_jobs(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: False)
    # kps_extract is torch-independent (InsightFace SCRFD), so it is advertised whenever
    # that backend is installed — force it off here so this torch-gating assertion stays
    # deterministic whether or not insightface is present in the env (sc-4433).
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu", "nvidia"]})

    # lora_train dry-run validation needs no inference backend, so it is
    # advertised even without torch; generation job types are not.
    assert capabilities == ["gpu", "lora_train", "nvidia"]
    for job_type in ("image_generate", "image_edit", "image_vqa", "video_generate", "training_caption", "prompt_refine"):
        assert job_type not in capabilities


def test_gpu_worker_advertises_lora_train_without_inference_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})

    assert "lora_train" in capabilities


def test_cpu_worker_does_not_advertise_lora_train():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert "lora_train" not in capabilities


def test_python_worker_only_advertises_inference_job_capabilities(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    # Detection backends (pose/kps) are advertised only when their optional deps are
    # installed; force kps off so this torch-inference list stays deterministic whether
    # or not insightface/cv2 happen to be present in the test env (sc-4433).
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    job_capabilities = [capability for capability in capabilities if capability != "gpu"]

    assert job_capabilities == [
        "image_detail",
        "image_edit",
        "image_generate",
        "image_interleave",
        "image_upscale",
        "image_vqa",
        "lora_train",
        "lora_train_execute",
        "person_replace",
        "prompt_refine",
        "training_caption",
        "video_bridge",
        "video_extend",
        "video_generate",
    ]


def test_run_image_upscale_writes_single_child_asset(tmp_path, monkeypatch):
    # sc-2431: standalone upscale of an existing asset. Mock the engine so the
    # orchestration (source resolve → one child asset + lineage) is exercised
    # without torch/weights (CI has neither).
    from scene_worker.image_adapters import run_image_upscale

    source = Image.new("RGB", (8, 6), "blue")
    source_path = tmp_path / "src.png"
    source.save(source_path, "PNG")

    class _FakeUpscaler:
        id = "real-esrgan"

        def upscale(self, image, *, request, cancel_requested):
            factor = request.upscale.factor
            return image.resize((image.width * factor, image.height * factor))

    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: source_path,
    )
    monkeypatch.setattr(
        "scene_worker.image_adapters.create_image_upscaler",
        lambda request, **kwargs: _FakeUpscaler(),
    )

    job = {
        "id": "job_upscale_1",
        "payload": {
            "projectId": "project_1",
            "sourceAssetId": "asset_src",
            "factor": 2,
            "engine": "real-esrgan",
            "displayName": "Studio still",
        },
    }
    result = run_image_upscale(
        SimpleNamespace(gpu_id=None),
        job,
        project_path=tmp_path,
        progress=lambda *args: None,
        cancel_requested=lambda: False,
    )

    writes = result["assetWrites"]
    assert len(writes) == 1
    fact = writes[0]
    # Upscaled to 2x of the 8x6 source.
    assert (fact["width"], fact["height"]) == (16, 12)
    assert fact["mode"] == "image_upscale"
    assert fact["sourceAssetId"] == "asset_src"
    assert fact["parents"] == ["asset_src"]
    assert fact["displayName"] == "Studio still (2x upscaled)"
    assert fact["extra"] == {
        "isUpscaled": True,
        "upscaledFromAssetId": "asset_src",
        "factor": 2,
        "engine": "real-esrgan",
    }
    # The upscaled PNG was actually written to the reported path at 2x size.
    written_path = tmp_path / fact["mediaPath"]
    assert written_path.is_file()
    assert Image.open(written_path).size == (16, 12)
    # One generation set wrapping the single child asset.
    assert result["expectedCount"] == 1
    assert result["generationSet"]["id"] == result["generationSetId"]


def test_image_request_threads_mask_asset_id(monkeypatch):
    # sc-2476: an edit job can carry an inpaint mask alongside the source.
    request = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "maskAssetId": "asset_mask"}}
    )
    assert request.mask_asset_id == "asset_mask"
    # Absent maskAssetId → None (whole-image edit).
    plain = image_request_from_job({"payload": {"projectId": "p", "sourceAssetId": "asset_src"}})
    assert plain.mask_asset_id is None


def test_model_supports_inpaint_matrix():
    # sc-2476: SDXL family honors a mask; instruction-edit / img2img-only models don't.
    assert model_supports_inpaint("sdxl") is True
    assert model_supports_inpaint("realvisxl") is True
    assert model_supports_inpaint("qwen_image_edit_2511") is False
    assert model_supports_inpaint("z_image_edit") is False
    assert model_supports_inpaint("unknown_model") is False


def test_load_mask_image_resolves_grayscale_and_aligns_to_output(tmp_path, monkeypatch):
    # sc-2476: the mask resolves only through the project sidecar (no path escape) and
    # is returned grayscale, resized to the output W×H so it aligns with the source.
    mask = Image.new("RGB", (40, 30), "white")
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path, "PNG")
    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: mask_path,
    )
    request = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "maskAssetId": "asset_mask", "width": 512, "height": 256}}
    )
    loaded = load_mask_image(tmp_path, request)
    assert loaded.mode == "L"
    assert loaded.size == (512, 256)

    # No mask id → explicit error (callers gate on mask_asset_id first).
    with pytest.raises(RuntimeError, match="mask image asset"):
        load_mask_image(tmp_path, image_request_from_job({"payload": {"projectId": "p"}}))


def test_normalize_fit_mode_defaults_to_crop():
    # sc-2552: missing/unknown fit modes fall back to crop (never silently stretch).
    assert normalize_fit_mode(None) == "crop"
    assert normalize_fit_mode("") == "crop"
    assert normalize_fit_mode("bogus") == "crop"
    # Known modes pass through, case-insensitively.
    assert normalize_fit_mode("crop") == "crop"
    assert normalize_fit_mode("pad") == "pad"
    assert normalize_fit_mode("outpaint") == "outpaint"
    assert normalize_fit_mode("Stretch") == "stretch"
    # Default on the request model is crop when fitMode is absent.
    assert image_request_from_job({"payload": {"projectId": "p"}}).fit_mode == "crop"
    assert (
        image_request_from_job({"payload": {"projectId": "p", "fitMode": "pad"}}).fit_mode == "pad"
    )


def test_fit_image_crop_pad_stretch_geometry():
    # sc-2552: a 2:1 source fitted to a 1:1 box. crop covers (no bars), pad letterboxes
    # (neutral bars), stretch distorts. All return exactly the target size.
    src = Image.new("RGB", (40, 20), (255, 0, 0))  # solid red, landscape 2:1

    cropped = fit_image(src, 100, 100, "crop")
    assert cropped.size == (100, 100)
    # Cover scale (×5) fills the frame with red — no bars anywhere.
    assert cropped.getpixel((50, 0)) == (255, 0, 0)
    assert cropped.getpixel((50, 99)) == (255, 0, 0)

    padded = fit_image(src, 100, 100, "pad")
    assert padded.size == (100, 100)
    # Contain scale (×2.5) → 100×50 centered; top/bottom rows are neutral bars.
    assert padded.getpixel((50, 0)) == (0, 0, 0)
    assert padded.getpixel((50, 99)) == (0, 0, 0)
    assert padded.getpixel((50, 50)) == (255, 0, 0)

    # outpaint shares pad geometry (the bars become the generate region downstream).
    assert fit_image(src, 100, 100, "outpaint").getpixel((50, 0)) == (0, 0, 0)

    stretched = fit_image(src, 100, 100, "stretch")
    assert stretched.size == (100, 100)
    assert stretched.getpixel((50, 0)) == (255, 0, 0)  # distorted to fill, no bars


def test_load_source_image_honors_fit_mode_and_size_override(tmp_path, monkeypatch):
    # sc-2552: the source is fitted per request.fit_mode; explicit width/height override
    # the request dims (SenseNova snapped-bucket path).
    src = Image.new("RGB", (40, 20), (255, 0, 0))
    src_path = tmp_path / "src.png"
    src.save(src_path, "PNG")
    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: src_path,
    )
    # Output dims go through the 256-floor clamp in image_request_from_job, so use 400².
    pad_req = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "fitMode": "pad", "width": 400, "height": 400}}
    )
    padded = load_source_image(tmp_path, pad_req)
    assert padded.size == (400, 400)
    assert padded.getpixel((200, 0)) == (0, 0, 0)  # letterbox bar
    assert padded.getpixel((200, 200)) == (255, 0, 0)  # centered image

    # Default (crop) covers the frame — no bars.
    crop_req = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "width": 400, "height": 400}}
    )
    assert load_source_image(tmp_path, crop_req).getpixel((200, 0)) == (255, 0, 0)

    # Size override wins over the request dims (SenseNova snapped bucket); bypasses clamp.
    assert load_source_image(tmp_path, crop_req, 64, 48).size == (64, 48)


def test_load_mask_image_pad_keeps_bars_black(tmp_path, monkeypatch):
    # sc-2552: a padded mask must keep the bars black (= keep), only the original
    # region carries the user's white edit area, aligned with the padded source.
    mask = Image.new("RGB", (40, 20), "white")
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path, "PNG")
    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: mask_path,
    )
    request = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "maskAssetId": "asset_mask", "fitMode": "pad", "width": 400, "height": 400}}
    )
    loaded = load_mask_image(tmp_path, request)
    assert loaded.mode == "L"
    assert loaded.size == (400, 400)
    assert loaded.getpixel((200, 0)) == 0  # padded bar = keep
    assert loaded.getpixel((200, 200)) == 255  # original region = edit


def test_outpaint_border_mask_marks_bars_white_keeps_center_black():
    # sc-2553: white = padded border to generate, black = centered source to keep,
    # geometry matching fit_image(..., "pad").
    src = Image.new("RGB", (40, 20), (255, 0, 0))  # 2:1 → top/bottom bars in a square box
    mask = outpaint_border_mask(src, 400, 400)
    assert mask.mode == "L"
    assert mask.size == (400, 400)
    assert mask.getpixel((200, 0)) == 255  # top bar = generate
    assert mask.getpixel((200, 399)) == 255  # bottom bar = generate
    assert mask.getpixel((200, 200)) == 0  # centered source = keep
    # A source already matching the box aspect leaves nothing to generate (all keep).
    flush = outpaint_border_mask(Image.new("RGB", (50, 50)), 400, 400)
    assert flush.getpixel((0, 0)) == 0 and flush.getpixel((200, 200)) == 0


def test_outpaint_border_mask_feather_softens_seam():
    # sc-2553: a feather introduces intermediate (anti-aliased) values at the seam;
    # the hard mask is strictly binary.
    src = Image.new("RGB", (40, 20), (255, 0, 0))
    hard = outpaint_border_mask(src, 400, 400, feather=0)
    soft = outpaint_border_mask(src, 400, 400, feather=20)
    column_hard = {hard.getpixel((200, y)) for y in range(400)}
    assert column_hard <= {0, 255}  # binary
    assert any(0 < soft.getpixel((200, y)) < 255 for y in range(400))  # gradient


def test_outpaint_feather_default_override_and_clamp():
    # sc-2553: feather defaults to ~1.5% of the short edge; advanced.outpaintFeather
    # overrides, clamped to [0, 128], bad values fall back to the default.
    base = {"projectId": "p", "mode": "edit_image", "sourceAssetId": "s",
            "width": 1024, "height": 768}
    default = round(768 * 0.015)
    assert SdxlDiffusersAdapter._outpaint_feather(image_request_from_job({"payload": base})) == default
    over = image_request_from_job({"payload": {**base, "advanced": {"outpaintFeather": 4}}})
    assert SdxlDiffusersAdapter._outpaint_feather(over) == 4
    huge = image_request_from_job({"payload": {**base, "advanced": {"outpaintFeather": 9999}}})
    assert SdxlDiffusersAdapter._outpaint_feather(huge) == 128
    bad = image_request_from_job({"payload": {**base, "advanced": {"outpaintFeather": "nope"}}})
    assert SdxlDiffusersAdapter._outpaint_feather(bad) == default


def test_run_image_upscale_requires_source_asset(monkeypatch, tmp_path):
    from scene_worker.image_adapters import run_image_upscale

    with pytest.raises(RuntimeError, match="source image asset"):
        run_image_upscale(
            SimpleNamespace(gpu_id=None),
            {"id": "job_x", "payload": {"projectId": "p"}},
            project_path=tmp_path,
            progress=lambda *args: None,
            cancel_requested=lambda: False,
        )


def test_model_supports_detail_matrix():
    # sc-2438: tile-ControlNet detail refine is SDXL-family only.
    assert model_supports_detail("sdxl") is True
    assert model_supports_detail("realvisxl") is True
    assert model_supports_detail("qwen_image_edit_2511") is False
    assert model_supports_detail("unknown_model") is False


def test_run_image_detail_writes_single_child_asset(tmp_path, monkeypatch):
    # sc-2438: standalone tile-ControlNet detail refine of an existing asset. Mock the
    # pipeline load + the tiled refine so the orchestration (source resolve → one child
    # asset + lineage + fact shape) runs without torch/diffusers/weights (CI has none).
    from scene_worker.image_adapters import run_image_detail

    source = Image.new("RGB", (12, 9), "green")
    source_path = tmp_path / "src.png"
    source.save(source_path, "PNG")
    refined = Image.new("RGB", (12, 9), "white")

    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: source_path,
    )
    monkeypatch.setattr(
        "scene_worker.image_adapters._load_detail_pipeline",
        lambda settings, **kwargs: object(),
    )
    monkeypatch.setattr(
        "scene_worker.image_adapters._refine_tiled_detail",
        lambda pipe, image, **kwargs: (refined, 1),
    )

    job = {
        "id": "job_detail_1",
        "payload": {
            "projectId": "project_1",
            "sourceAssetId": "asset_src",
            "model": "realvisxl",
            "displayName": "Studio still",
            "advanced": {"strength": 0.6, "cnScale": 0.8},
        },
    }
    result = run_image_detail(
        SimpleNamespace(gpu_id=None),
        job,
        project_path=tmp_path,
        progress=lambda *args: None,
        cancel_requested=lambda: False,
    )

    writes = result["assetWrites"]
    assert len(writes) == 1
    fact = writes[0]
    # Detail refine preserves dimensions (refines in place — unlike upscale).
    assert (fact["width"], fact["height"]) == (12, 9)
    assert fact["mode"] == "image_detail"
    assert fact["model"] == "realvisxl"
    assert fact["sourceAssetId"] == "asset_src"
    assert fact["parents"] == ["asset_src"]
    assert fact["displayName"] == "Studio still (detail enhanced)"
    assert fact["extra"] == {
        "isDetailEnhanced": True,
        "detailFromAssetId": "asset_src",
        "backbone": "realvisxl",
        "strength": 0.6,
        "cnScale": 0.8,
    }
    written_path = tmp_path / fact["mediaPath"]
    assert written_path.is_file()
    assert Image.open(written_path).size == (12, 9)
    assert result["expectedCount"] == 1
    assert result["generationSet"]["id"] == result["generationSetId"]


def test_run_image_detail_requires_source_asset(tmp_path):
    from scene_worker.image_adapters import run_image_detail

    with pytest.raises(RuntimeError, match="source image asset"):
        run_image_detail(
            SimpleNamespace(gpu_id=None),
            {"id": "job_x", "payload": {"projectId": "p"}},
            project_path=tmp_path,
            progress=lambda *args: None,
            cancel_requested=lambda: False,
        )


def test_run_image_detail_rejects_non_detail_model(tmp_path):
    # A non-SDXL model is rejected before any pipeline load (no mocking needed).
    from scene_worker.image_adapters import run_image_detail

    with pytest.raises(RuntimeError, match="does not support detail"):
        run_image_detail(
            SimpleNamespace(gpu_id=None),
            {"id": "job_x", "payload": {"projectId": "p", "sourceAssetId": "asset_src",
                                        "model": "qwen_image_edit_2511"}},
            project_path=tmp_path,
            progress=lambda *args: None,
            cancel_requested=lambda: False,
        )


def test_gpu_worker_advertises_lora_train_execute_only_with_inference_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    with_backend = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "lora_train" in with_backend
    assert "lora_train_execute" in with_backend

    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: False)
    without_backend = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    # Dry-run validation stays claimable; real execution is not advertised.
    assert "lora_train" in without_backend
    assert "lora_train_execute" not in without_backend


def test_python_cpu_worker_does_not_advertise_person_tracking_jobs():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert capabilities == ["cpu"]


def test_gpu_worker_advertises_real_person_jobs_when_backends_installed(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.detector_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.tracker_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.segmenter_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "person_detect" in capabilities
    assert "person_track" in capabilities
    assert "person_segment" in capabilities


def test_gpu_worker_advertises_pose_detect_when_backend_installed(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "pose_detect" in capabilities


def test_gpu_worker_omits_pose_detect_without_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "pose_detect" not in capabilities


def test_gpu_worker_advertises_kps_extract_when_backend_installed(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "kps_extract" in capabilities


def test_gpu_worker_omits_kps_extract_without_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "kps_extract" not in capabilities


def test_cpu_worker_never_advertises_pose_detect(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})
    assert capabilities == ["cpu"]


def test_gpu_worker_omits_person_jobs_without_detector_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.detector_backend_available", lambda: False)
    monkeypatch.setattr("scene_worker.runtime.tracker_backend_available", lambda: False)
    monkeypatch.setattr("scene_worker.runtime.segmenter_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "person_detect" not in capabilities
    assert "person_track" not in capabilities


def test_cpu_worker_never_advertises_real_person_jobs_even_with_backends(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.detector_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.tracker_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})
    assert capabilities == ["cpu"]


def test_python_cpu_child_disables_cuda():
    env = child_environment(SimpleNamespace(), worker_id="python-inference-worker-cpu", gpu_id="cpu")

    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_python_gpu_child_selects_cuda_device():
    env = child_environment(SimpleNamespace(), worker_id="worker-gpu-0", gpu_id="0")

    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_select_torch_device_uses_assigned_gpu_when_multiple_cuda_devices_are_visible():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 2

        class backends:
            mps = None

    assert select_torch_device(Torch, "1") == "cuda:1"


def test_select_torch_device_uses_visible_cuda_default_when_child_process_is_narrowed():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

        class backends:
            mps = None

    assert select_torch_device(Torch, "1") == "cuda"


def test_ltx_mps_gating_leaves_cuda_path_untouched():
    gating = ltx_mps_gating(cuda_available=True, device_str="cuda")
    assert gating == {
        "device": None,
        "disable_fp8": False,
        "force_offload_none": False,
        "fp32_audio": False,
        "guard_cuda_sync": False,
    }


def test_ltx_mps_gating_steers_apple_silicon_to_mps_recipe():
    gating = ltx_mps_gating(cuda_available=False, device_str="mps")
    assert gating == {
        "device": "mps",
        "disable_fp8": True,
        "force_offload_none": True,
        "fp32_audio": True,
        "guard_cuda_sync": True,
    }


def test_ltx_mps_gating_disables_cuda_features_on_cpu_without_forcing_mps_device():
    # A CPU host off CUDA still must drop fp8/offload (both CUDA-only) and guard the
    # unguarded cuda.synchronize, but it must not claim an mps device it does not have.
    gating = ltx_mps_gating(cuda_available=False, device_str="cpu")
    assert gating["device"] is None
    assert gating["disable_fp8"] is True
    assert gating["force_offload_none"] is True
    assert gating["guard_cuda_sync"] is True


def test_gpu_worker_fails_fast_when_torch_cuda_is_unavailable():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        require_inference_backend_for_gpu_worker(Torch, "0")


def test_auto_gpu_worker_fails_fast_when_inference_backend_is_unavailable():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            mps = None

    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        require_inference_backend_for_gpu_worker(Torch, "auto")


def test_mps_worker_can_advertise_generation_capabilities(monkeypatch):
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return True

    monkeypatch.setattr("scene_worker.image_adapters.importlib.import_module", lambda name: Torch if name == "torch" else None)

    capabilities = worker_capabilities({"id": "mps", "name": "Apple GPU", "capabilities": ["placeholder", "gpu"]})

    assert "image_generate" in capabilities
    assert "video_generate" in capabilities


def test_gpu_worker_accepts_mps_inference_backend():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return True

    require_inference_backend_for_gpu_worker(Torch, "mps")


def test_loaded_models_are_collected_from_adapter_cache():
    class Adapter:
        def loaded_models(self):
            return ["Tongyi-MAI/Z-Image-Turbo"]

    assert loaded_models_from_adapters({"z": Adapter()}) == ["Tongyi-MAI/Z-Image-Turbo"]


def test_z_image_loaded_models_include_repo_and_model_id():
    adapter = ZImageDiffusersAdapter()
    adapter._loaded_repo = "Tongyi-MAI/Z-Image-Turbo"
    adapter._loaded_model = "z_image_turbo"

    assert set(adapter.loaded_models()) == {"Tongyi-MAI/Z-Image-Turbo", "z_image_turbo"}


def test_z_image_turbo_model_target_defaults_and_guidance_helper():
    target = MODEL_TARGETS["z_image_turbo"]
    assert target["adapter"] == "z_image_diffusers"
    assert target["family"] == "z-image"
    assert target["steps"] == 8
    assert target["guidanceScale"] == 1.0

    adapter = ZImageDiffusersAdapter()
    assert adapter._guidance_scale(SimpleNamespace(model="z_image_turbo", advanced={})) == 1.0
    assert adapter._guidance_scale(SimpleNamespace(model="z_image_turbo", advanced={"guidanceScale": 0.0})) == 0.0
    assert adapter._guidance_scale(SimpleNamespace(model="z_image_turbo", advanced={"guidanceScale": "bad"})) == 1.0


def test_qwen_loaded_models_track_text_and_edit_repos_independently():
    adapter = QwenImageAdapter()
    adapter._text_repo = "Qwen/Qwen-Image"
    adapter._edit_repo = "Qwen/Qwen-Image-Edit"
    adapter._loaded_model = "qwen_image_edit"

    assert set(adapter.loaded_models()) == {"Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit", "qwen_image_edit"}


def test_qwen_image_edit_2511_model_target_defaults():
    target = MODEL_TARGETS["qwen_image_edit_2511"]
    assert target["adapter"] == "qwen_image"
    assert target["family"] == "qwen-image"
    assert target["supportsEdit"] is True
    # Model card: 40 steps, guidanceScale 1.0, trueCfgScale 4.0 (default).
    assert target["steps"] == 40
    assert target["guidanceScale"] == 1.0
    assert target["repo"] == "Qwen/Qwen-Image-Edit-2511"


def test_qwen_image_edit_legacy_ids_alias_to_2511_repo():
    # sc-2160: the August qwen_image_edit and the September qwen_image_edit_2509
    # IDs now route to the December 2511 weights so old jobs/presets resolve.
    legacy = MODEL_TARGETS["qwen_image_edit"]
    sept = MODEL_TARGETS["qwen_image_edit_2509"]
    assert legacy["repo"] == "Qwen/Qwen-Image-Edit-2511"
    assert sept["repo"] == "Qwen/Qwen-Image-Edit-2511"
    assert legacy["steps"] == 40
    assert sept["steps"] == 40
    assert legacy["guidanceScale"] == 1.0
    assert sept["guidanceScale"] == 1.0


def test_qwen_image_edit_2511_lightning_model_target_defaults():
    target = MODEL_TARGETS["qwen_image_edit_2511_lightning"]
    assert target["adapter"] == "qwen_image"
    assert target["family"] == "qwen-image"
    assert target["supportsEdit"] is True
    # 4-step distill: cfg 1.0, true_cfg_scale 1.0, shares 2511 base weights.
    assert target["steps"] == 4
    assert target["guidanceScale"] == 1.0
    assert target["trueCfgScale"] == 1.0
    assert target["repo"] == "Qwen/Qwen-Image-Edit-2511"
    assert target["distillLora"] == {
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    }


def test_create_image_adapter_routes_qwen_image_edit_2511():
    # All Qwen Edit IDs ride the same QwenImageAdapter; pipeline-class selection
    # happens inside _load_pipeline based on the model id.
    adapter = create_image_adapter({"payload": {"model": "qwen_image_edit_2511"}})
    assert adapter.__class__.__name__ == "QwenImageAdapter"
    assert adapter.id == "qwen_image"
    lightning = create_image_adapter({"payload": {"model": "qwen_image_edit_2511_lightning"}})
    assert lightning.__class__.__name__ == "QwenImageAdapter"


def test_qwen_edit_pipeline_name_by_model():
    # sc-2160: every edit ID (incl. legacy aliases) ships the multi-image Plus
    # pipeline now that they all run against the 2511 base weights.
    assert QwenImageAdapter._edit_pipeline_name("qwen_image_edit") == "QwenImageEditPlusPipeline"
    assert QwenImageAdapter._edit_pipeline_name("qwen_image_edit_2509") == "QwenImageEditPlusPipeline"
    assert QwenImageAdapter._edit_pipeline_name("qwen_image_edit_2511") == "QwenImageEditPlusPipeline"
    assert QwenImageAdapter._edit_pipeline_name("qwen_image_edit_2511_lightning") == "QwenImageEditPlusPipeline"
    # Defensive: anything else still gets the single-image pipeline, matching
    # the same edit-style API surface.
    assert QwenImageAdapter._edit_pipeline_name("qwen_image") == "QwenImageEditPipeline"


def test_qwen_distill_lora_helpers():
    base_target = MODEL_TARGETS["qwen_image_edit_2511"]
    lightning_target = MODEL_TARGETS["qwen_image_edit_2511_lightning"]
    assert QwenImageAdapter._distill_lora_for(base_target) is None
    assert QwenImageAdapter._distill_key_for(None) is None
    spec = QwenImageAdapter._distill_lora_for(lightning_target)
    assert spec is not None
    key = QwenImageAdapter._distill_key_for(spec)
    assert key == "lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def test_qwen_guidance_scale_reads_model_target_default():
    # 2511 family defaults to guidanceScale 1.0 (Plus pipeline requirement).
    request = SimpleNamespace(model="qwen_image_edit_2511", advanced={})
    assert QwenImageAdapter()._guidance_scale(request) == 1.0
    # qwen_image (text-to-image) still defaults to 4.0.
    request_t2i = SimpleNamespace(model="qwen_image", advanced={})
    assert QwenImageAdapter()._guidance_scale(request_t2i) == 4.0
    # User override wins.
    request_override = SimpleNamespace(model="qwen_image_edit_2511", advanced={"guidanceScale": 3.5})
    assert QwenImageAdapter()._guidance_scale(request_override) == 3.5


def test_qwen_true_cfg_scale_default_per_model():
    adapter = QwenImageAdapter()
    # Base family: 4.0 default.
    request = SimpleNamespace(model="qwen_image_edit_2511", advanced={})
    assert adapter._true_cfg_scale_default(request) == 4.0
    # Lightning: distilled at 1.0.
    lightning_request = SimpleNamespace(model="qwen_image_edit_2511_lightning", advanced={})
    assert adapter._true_cfg_scale_default(lightning_request) == 1.0
    # User override wins on either.
    override = SimpleNamespace(model="qwen_image_edit_2511_lightning", advanced={"trueCfgScale": 2.0})
    assert adapter._true_cfg_scale_default(override) == 2.0


def test_qwen_use_reference_only_for_character_image_with_reference():
    use = QwenImageAdapter._use_reference
    # Character Studio reference path requires both mode + a reference asset.
    assert use(SimpleNamespace(mode="character_image", reference_asset_id="a")) is True
    # No reference id → no character_image route (the picker shouldn't allow this
    # combination, but the worker double-checks).
    assert use(SimpleNamespace(mode="character_image", reference_asset_id=None)) is False
    # Edit and text-to-image modes go through their own paths regardless of
    # whether a reference id is also present.
    assert use(SimpleNamespace(mode="edit_image", reference_asset_id="a")) is False
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id="a")) is False


def test_qwen_reference_true_cfg_scale_default_and_clamp():
    adapter = QwenImageAdapter()
    # Model-card default 4.0; clamp [1, 10] (below 1 disables CFG and the edit
    # pipeline needs it > 1, above 10 collapses to pure negative-prompt steering).
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={})) == 4.0
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 2.5})) == 2.5
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 0.5})) == 1.0
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 99})) == 10.0
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": "x"})) == 4.0


def test_qwen_reference_run_pipeline_passes_image_and_true_cfg(tmp_path, monkeypatch):
    """A character_image job with referenceAssetId drives the reference branch of
    _run_pipeline: load_reference_image(project_path, reference_asset_id) → image=
    kwarg, plus true_cfg_scale + negative_prompt (defaulted to ' ' so true CFG
    engages on Qwen edit pipelines). Mirrors the FLUX/SDXL torch-free pattern."""

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.last_kwargs: dict[str, Any] = {}

        # Named params so filter_call_kwargs keeps the Qwen-specific kwargs we
        # need to assert (FakePipe's `__call__` introspects clean via signature).
        def __call__(
            self,
            *,
            prompt=None,
            negative_prompt=None,
            image=None,
            true_cfg_scale=None,
            height=None,
            width=None,
            num_inference_steps=None,
            guidance_scale=None,
            generator=None,
            **kwargs,
        ):
            self.last_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "true_cfg_scale": true_cfg_scale,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                **kwargs,
            }
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen: list[tuple] = []

    def fake_load_reference_image(project_path, reference_asset_id):
        seen.append((project_path, reference_asset_id))
        return FakeImage()

    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image", fake_load_reference_image
    )

    # sampler_selection_from_advanced + apply_sampler poke the pipe — short-circuit.
    monkeypatch.setattr(
        "scene_worker.image_adapters.apply_sampler",
        lambda *args, **kwargs: None,
    )

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "character_image",
                "model": "qwen_image_edit_2509",
                "prompt": "the same character at a cafe",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {"trueCfgScale": 3.0},
            }
        }
    )
    pipe = FakePipe()
    result = QwenImageAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"), pipe, request, 7, project_path
    )
    # Reference branch ran: load_reference_image called, image= kwarg threaded
    # through with the loaded reference, true_cfg_scale from advanced overrides
    # default, and negative_prompt defaulted to " " so true CFG engages (Qwen's
    # edit pipeline needs a non-empty negative prompt for guidance > 1).
    assert seen == [(project_path, "asset-ref")]
    assert pipe.last_kwargs["true_cfg_scale"] == 3.0
    assert pipe.last_kwargs["negative_prompt"] == " "
    assert pipe.last_kwargs["image"] is not None
    assert not isinstance(pipe.last_kwargs["image"], list)  # single reference, no pose
    assert result is FakeOutput.images[0]


def test_qwen_reference_run_pipeline_threads_pose_skeleton_as_second_image(tmp_path, monkeypatch):
    """Best-effort pose tier (sc-2256): when _run_pipeline gets a pose_skeleton,
    the reference branch passes image=[reference, skeleton] to the multi-image
    edit pipeline (so the skeleton steers the body pose) and the prompt carries
    the pose cue. Without a skeleton it's a single-image reference (covered
    above)."""
    from scene_worker.character_studio_angles import POSE_SKELETON_PROMPT

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.last_kwargs: dict[str, Any] = {}

        def __call__(self, *, prompt=None, image=None, **kwargs):
            self.last_kwargs = {"prompt": prompt, "image": image, **kwargs}
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    reference_sentinel = FakeImage()
    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image",
        lambda project_path, reference_asset_id: reference_sentinel,
    )
    monkeypatch.setattr("scene_worker.image_adapters.apply_sampler", lambda *a, **k: None)

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "character_image",
                "model": "qwen_image_edit_2511_lightning",
                "prompt": "the same character",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {},
            }
        }
    )
    skeleton_sentinel = FakeImage()
    pipe = FakePipe()
    QwenImageAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"),
        pipe,
        request,
        7,
        project_path,
        prompt_override="the same character, " + POSE_SKELETON_PROMPT,
        pose_skeleton=skeleton_sentinel,
    )
    # Multi-image: reference first (identity), skeleton second (pose target).
    assert pipe.last_kwargs["image"] == [reference_sentinel, skeleton_sentinel]
    assert POSE_SKELETON_PROMPT in pipe.last_kwargs["prompt"]


def test_augment_prompt_for_pose_appends_skeleton_cue():
    from scene_worker.character_studio_angles import POSE_SKELETON_PROMPT, augment_prompt_for_pose

    assert augment_prompt_for_pose("a woman at a cafe") == f"a woman at a cafe, {POSE_SKELETON_PROMPT}"
    # Empty base → just the cue (caller still gets a usable instruction).
    assert augment_prompt_for_pose("") == POSE_SKELETON_PROMPT
    assert augment_prompt_for_pose("  a woman.  ") == f"a woman, {POSE_SKELETON_PROMPT}"


def test_qwen_strict_pose_entries_gating():
    """sc-2291: the base-Qwen strict pose path engages for a target that declares
    controlNetPose (base qwen_image) with advanced.poses — NOT for edit jobs, NOT for
    edit targets (no controlNetPose), and (unlike Kolors) with NO reference required."""
    from scene_worker.image_adapters import MODEL_TARGETS, QwenImageAdapter

    entries = QwenImageAdapter._strict_pose_entries
    base = MODEL_TARGETS["qwen_image"]  # declares controlNetPose
    edit = MODEL_TARGETS["qwen_image_edit_2511"]  # no controlNetPose → best-effort tier
    kp = [{"id": "sit", "keypoints": [[0.5, 0.5]] * 18}]
    # Base target + poses → fires, with OR without a reference (pose-from-prompt).
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={"poses": kp}), base) == kp
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id=None, advanced={"poses": kp}), base) == kp
    # Edit mode never takes the strict path.
    assert entries(SimpleNamespace(mode="edit_image", reference_asset_id="r", advanced={"poses": kp}), base) == []
    # Edit target (no controlNetPose) stays on the best-effort tier (sc-2256).
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={"poses": kp}), edit) == []
    # No poses → no strict path.
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={}), base) == []


def test_qwen_strict_pose_set_loops_poses_with_shared_seed(tmp_path, monkeypatch):
    """sc-2291: generate() on base qwen_image with advanced.poses routes to the strict
    ControlNet pose path — one image per pose, a single shared seed, hands/face threaded
    through, and poseLibrary + controlNetPose recorded in settings."""
    from scene_worker import image_adapters as ia

    monkeypatch.setattr(ia.QwenImageAdapter, "_load_pose_pipeline", lambda self, *a, **k: object())
    monkeypatch.setattr(ia.QwenImageAdapter, "_apply_loras", lambda self, *a, **k: None)
    monkeypatch.setattr(ia, "select_torch_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: SimpleNamespace() if name == "torch" else importlib.import_module(name),
    )

    calls: list[dict] = []

    def fake_run_pose(self, settings, pipe, request, seed, project_path, keypoints, hands, face, cancel_requested=None):
        from PIL import Image as _Image
        calls.append({"seed": seed, "keypoints": keypoints, "hands": hands, "face": face})
        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(ia.QwenImageAdapter, "_run_pose", fake_run_pose)

    captured: dict = {}

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, raw_settings, **kwargs):
            captured["raw_settings"] = raw_settings
            captured["image_count"] = image_count
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)

    kp = [[0.5, 0.1 + 0.04 * i] for i in range(18)]
    hands = [[[0.4, 0.4]] * 21, [[0.6, 0.4]] * 21]
    face = [[0.5, 0.3]] * 68
    job = {"id": "job_qwen_pose", "payload": {
        "projectId": "p", "mode": "character_image", "model": "qwen_image", "prompt": "the character",
        "count": 1, "width": 64, "height": 64,
        "advanced": {"poses": [
            {"id": "sit_01", "keypoints": kp},  # body-only
            {"id": "dance_01", "keypoints": kp, "hands": hands, "face": face},  # whole-body
        ]},
    }}
    ia.QwenImageAdapter().generate(
        settings=SimpleNamespace(gpu_id="cpu"), job=job, request=image_request_from_job(job),
        project_path=tmp_path, progress=lambda *a, **k: None, cancel_requested=lambda: False,
    )
    assert captured["image_count"] == 2  # one per pose, not request.count
    assert len(calls) == 2
    assert calls[0]["seed"] == calls[1]["seed"]  # shared seed across the set
    # Whole-body hands/face thread through to the DWPose-trained Qwen CN; body-only → None.
    assert calls[0]["hands"] is None and calls[0]["face"] is None
    assert calls[1]["hands"] is not None and len(calls[1]["face"]) == 68
    assert captured["raw_settings"].get("poseLibrary") is True
    assert captured["raw_settings"].get("controlNetPose") == "InstantX/Qwen-Image-ControlNet-Union"


def test_qwen_run_pose_conditions_controlnet_without_reference(tmp_path, monkeypatch):
    """sc-2291 strict tier: _run_pose feeds the rendered DWPose skeleton as control_image
    and controlScale as controlnet_conditioning_scale — pure txt2img + control, with NO
    reference / ip_adapter / img2img `image` kwarg (pose-from-prompt)."""
    import numpy as _np
    from scene_worker import image_adapters as ia

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.last_kwargs: dict = {}

        def __call__(self, *, prompt=None, control_image=None, controlnet_conditioning_scale=None, **kwargs):
            self.last_kwargs = {
                "prompt": prompt, "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale, **kwargs,
            }
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr("scene_worker.image_adapters.apply_sampler", lambda *a, **k: None)
    skeleton_sentinel = FakeImage()
    monkeypatch.setattr(
        ia, "draw_wholebody", lambda w, h, kps, hands=None, face=None, stickwidth=4: _np.zeros((h, w, 3), dtype=_np.uint8)
    )
    monkeypatch.setattr("scene_worker.image_adapters.Image", SimpleNamespace(fromarray=lambda arr: skeleton_sentinel))

    request = image_request_from_job({"payload": {
        "projectId": "p", "mode": "character_image", "model": "qwen_image",
        "prompt": "the character", "width": 16, "height": 16, "count": 1,
        "advanced": {"controlScale": 0.85},
    }})
    pipe = FakePipe()
    kp = [(0.5, 0.1 + 0.04 * i) for i in range(18)]
    ia.QwenImageAdapter()._run_pose(SimpleNamespace(gpu_id="cpu"), pipe, request, 7, tmp_path, kp, None, None)
    assert pipe.last_kwargs["control_image"] is skeleton_sentinel  # pose → ControlNet
    assert pipe.last_kwargs["controlnet_conditioning_scale"] == 0.85  # advanced.controlScale
    # Pose-from-prompt: no reference / IP-Adapter / img2img image kwarg.
    assert "image" not in pipe.last_kwargs
    assert "ip_adapter_image" not in pipe.last_kwargs


def test_qwen_character_image_angle_set_applies_loras_once_then_loops_angles(tmp_path, monkeypatch):
    """sc-2225: a character_image + angleSet job on a diffusers backbone (Qwen) must
    apply request.loras exactly once, BEFORE the per-angle loop, and emit one image per
    canonical angle. Guards the regression that angle-set mode skips the LoRA merge."""

    class FakeImage:
        def convert(self, _mode):
            return self

    # The worker unit suite runs without torch installed; generate() does
    # `importlib.import_module("torch")`, so stand in a fake. With gpu_id="cpu",
    # select_torch_device returns early and never touches it.
    class FakeTorch:
        pass

    adapter = QwenImageAdapter()
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr("scene_worker.image_adapters.gpu_memory_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(adapter, "_load_pipeline", lambda *a, **k: object())

    apply_calls: list[list] = []
    monkeypatch.setattr(
        adapter, "_apply_loras", lambda pipe, request, lora_key=None: apply_calls.append(list(request.loras))
    )

    run_overrides: list = []

    # generate() always passes pose_skeleton (None for the angle path); accept it
    # here so the fake matches the real _run_pipeline signature (sc-2256).
    def fake_run(settings, pipe, request, seed, project_path, *, cancel_requested=None, prompt_override=None, pose_skeleton=None):
        # The LoRA merge must already have happened before any angle is generated.
        assert apply_calls, "loras must be applied before the angle loop runs"
        assert pose_skeleton is None  # angle path is prompt-driven, no skeleton
        run_overrides.append(prompt_override)
        return FakeImage()

    monkeypatch.setattr(adapter, "_run_pipeline", fake_run)

    captured: dict = {}

    def fake_writer(self, *, image_count, image_at_index, **kwargs):
        captured["image_count"] = image_count
        for index in range(image_count):
            image_at_index(index)
        return {"images": [], "count": image_count}

    monkeypatch.setattr(ImageAssetWriter, "write_incremental_outputs", fake_writer)

    loras = [{"id": "kelsie", "path": "/loras/kelsie.safetensors", "families": ["qwen-image"]}]
    job = {
        "id": "job-qwen-angle",
        "payload": {
            "projectId": "p",
            "mode": "character_image",
            "model": "qwen_image_edit_2511",
            "prompt": "the character",
            "referenceAssetId": "ref-1",
            "count": 1,
            "width": 16,
            "height": 16,
            "loras": loras,
            "advanced": {"angleSet": True},
        },
    }
    adapter.generate(
        settings=SimpleNamespace(gpu_id="cpu"),
        job=job,
        request=image_request_from_job(job),
        project_path=tmp_path,
        progress=lambda *a, **k: None,
        cancel_requested=lambda: False,
    )

    # Applied exactly once (per pipe, before the loop) with the requested loras.
    assert apply_calls == [loras]
    # One image per canonical angle, each with its own per-angle prompt augment.
    assert captured["image_count"] == len(CHARACTER_ANGLE_SET_ORDER)
    assert len(run_overrides) == len(CHARACTER_ANGLE_SET_ORDER)
    assert len(set(run_overrides)) == len(CHARACTER_ANGLE_SET_ORDER)


def test_filter_call_kwargs_preserves_none_for_accepted_parameters():
    assert filter_call_kwargs(AcceptsNone(), {"prompt": "city", "image": None, "extra": 1}) == {
        "prompt": "city",
        "image": None,
    }


def test_lora_loader_applies_weights_and_reuses_cached_state(tmp_path):
    first = tmp_path / "style.safetensors"
    second = tmp_path / "detail.safetensors"
    first.write_bytes(b"lora")
    second.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    loras = [
        {"id": "style", "installedPath": str(first), "weight": 0.5},
        {"id": "detail", "installedPath": str(second), "weight": 0.8},
    ]

    state = apply_loras_to_pipeline(pipe, loras, adapter_id="diffusers_test")
    same_state = apply_loras_to_pipeline(
        pipe,
        loras,
        adapter_id="diffusers_test",
        previous_state=state,
    )

    assert same_state == state
    assert [path for path, _name in pipe.loaded] == [str(first), str(second)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.5, 0.8])]
    assert pipe.unloaded == 0


def test_wan_moe_lora_loads_low_noise_into_transformer_2(tmp_path):
    # A trained Wan A14B MoE LoRA is a dir with a high/low pair (sc-1953). The
    # high-noise half loads into the transformer, the low-noise half into
    # transformer_2 via diffusers' load_into_transformer_2 (sc-1955).
    moe = tmp_path / "lora_moe"
    moe.mkdir()
    (moe / "char.high_noise.safetensors").write_bytes(b"lora")
    (moe / "char.low_noise.safetensors").write_bytes(b"lora")
    lora = {
        "id": "char",
        "installedPath": str(moe),
        "families": ["wan-video"],
        "baseModel": "wan_2_2_t2v_14b",
        "weight": 0.8,
    }

    specs = normalize_lora_specs([lora])
    assert specs[0].path.endswith("char.high_noise.safetensors")
    assert specs[0].secondary_path.endswith("char.low_noise.safetensors")

    pipe = FakeMoeLoraPipe()
    apply_loras_to_pipeline(
        pipe, [lora], adapter_id="wan_video", model_family="wan-video", model_id="wan_2_2_t2v_14b"
    )
    assert len(pipe.loaded) == 2
    high_path, high_name, high_t2 = pipe.loaded[0]
    low_path, low_name, low_t2 = pipe.loaded[1]
    assert high_path.endswith("char.high_noise.safetensors") and high_t2 is False
    assert low_path.endswith("char.low_noise.safetensors") and low_t2 is True
    # Both experts share the same adapter name so set_adapters activates both.
    assert high_name == low_name


def test_wan_moe_lora_on_dense_pipe_skips_second_expert(tmp_path):
    # A MoE LoRA on a pipe without transformer_2 loads only the high-noise half
    # (base-model gating blocks this combo upstream, but it must not crash).
    moe = tmp_path / "lora_moe2"
    moe.mkdir()
    (moe / "char.high_noise.safetensors").write_bytes(b"lora")
    (moe / "char.low_noise.safetensors").write_bytes(b"lora")
    pipe = FakeLoraPipe()  # dense: no transformer_2
    apply_loras_to_pipeline(
        pipe,
        [{"id": "char", "installedPath": str(moe), "families": ["wan-video"]}],
        adapter_id="wan_video",
        model_family="wan-video",
        model_id="wan_2_2",
    )
    assert len(pipe.loaded) == 1
    assert pipe.loaded[0][0].endswith("char.high_noise.safetensors")


def test_validate_lora_compatibility_gates_wan_base_model():
    wan_5b = {"id": "l", "families": ["wan-video"], "baseModel": "wan_2_2"}
    # A 5B LoRA on a 14B model is rejected (both family wan-video, incompatible arch).
    with pytest.raises(RuntimeError, match="not interchangeable"):
        validate_lora_compatibility(
            [wan_5b], model_family="wan-video", adapter_id="wan_video", model_id="wan_2_2_t2v_14b"
        )
    # Exact base-model match passes.
    validate_lora_compatibility(
        [wan_5b], model_family="wan-video", adapter_id="wan_video", model_id="wan_2_2"
    )
    # No recorded baseModel -> family gating only (legacy/imported), no rejection.
    validate_lora_compatibility(
        [{"id": "l2", "families": ["wan-video"]}],
        model_family="wan-video",
        adapter_id="wan_video",
        model_id="wan_2_2_t2v_14b",
    )
    # No model_id -> base-model gate is inert (back-compat with other call sites).
    validate_lora_compatibility([wan_5b], model_family="wan-video", adapter_id="wan_video")


def test_lora_loader_clears_previous_adapters_between_jobs(tmp_path):
    first = tmp_path / "style.safetensors"
    second = tmp_path / "detail.safetensors"
    first.write_bytes(b"lora")
    second.write_bytes(b"lora")
    pipe = FakeLoraPipe()

    state = apply_loras_to_pipeline(pipe, [{"id": "style", "installedPath": str(first)}], adapter_id="diffusers_test")
    apply_loras_to_pipeline(
        pipe,
        [{"id": "detail", "installedPath": str(second)}],
        adapter_id="diffusers_test",
        previous_state=state,
    )

    assert pipe.unloaded == 1
    assert pipe.loaded[-1][0] == str(second)


def test_adapter_network_type_defaults_to_lora_for_plain_file(tmp_path):
    plain = tmp_path / "plain.safetensors"
    plain.write_bytes(b"not a real safetensors header")
    # Unreadable/absent metadata resolves to lora — every legacy adapter is lora.
    assert adapter_network_type(plain) == "lora"


def test_reject_lokr_loras_raises_only_for_lokr(monkeypatch, tmp_path):
    lora_file = tmp_path / "a.safetensors"
    lora_file.write_bytes(b"x")
    lokr_file = tmp_path / "b.safetensors"
    lokr_file.write_bytes(b"x")
    types = {str(lora_file): "lora", str(lokr_file): "lokr"}
    monkeypatch.setattr(
        "scene_worker.lora_adapters.adapter_network_type", lambda path: types[str(path)]
    )
    lora_spec = LoraSpec(id="a", path=str(lora_file), weight=1.0, adapter_name="a")
    lokr_spec = LoraSpec(id="b", path=str(lokr_file), weight=1.0, adapter_name="b")

    reject_lokr_loras([lora_spec], "mlx_test")  # all-lora: no raise
    with pytest.raises(RuntimeError, match="LoKr"):
        reject_lokr_loras([lora_spec, lokr_spec], "mlx_test")


def test_apply_loras_routes_lokr_through_injection(monkeypatch, tmp_path):
    lokr_file = tmp_path / "char.safetensors"
    lokr_file.write_bytes(b"x")
    monkeypatch.setattr("scene_worker.lora_adapters.adapter_network_type", lambda path: "lokr")
    injected = []
    monkeypatch.setattr(
        "scene_worker.lora_adapters.inject_lokr_adapter",
        lambda pipe, spec, *, adapter_id: injected.append(spec.adapter_name),
    )
    pipe = FakeLokrPipe()

    state = apply_loras_to_pipeline(
        pipe,
        [{"id": "char", "installedPath": str(lokr_file), "weight": 0.7}],
        adapter_id="sdxl_test",
    )

    # LoKr never calls load_lora_weights; it injects, then sets weight on the module.
    assert pipe.loaded == []
    assert injected == list(state.adapter_names)
    assert len(state.adapter_names) == 1
    assert pipe.unet.set_calls[-1] == (list(state.adapter_names), [0.7])


def test_clear_loras_prefers_delete_adapters_so_lokr_is_removed():
    # delete_adapters removes injected LoKr adapters; unload_lora_weights (LoRA-only)
    # would leak them into the next job.
    pipe = FakeTargetedLoraPipe()
    clear_loras(pipe, ("char",), adapter_id="sdxl_test")
    assert pipe.deleted == [["char"]]
    assert pipe.unloaded == 0


def test_clear_loras_mixed_lycoris_and_peft_deletes_only_peft_names(monkeypatch):
    # sc-4181: on a cached pipeline holding a LyCORIS net AND a peft adapter,
    # clear_loras must pass only the non-LyCORIS leftovers to delete_adapters —
    # the full list includes the just-restored LyCORIS name, which diffusers'
    # delete_adapters rejects, failing the next job.
    monkeypatch.setattr(
        "scene_worker.lora_adapters._restore_lycoris_nets",
        lambda module, names: {"lyc_style"},
    )
    pipe = FakeTargetedLoraPipe()
    clear_loras(pipe, ("lyc_style", "char"), adapter_id="sdxl_test")
    assert pipe.deleted == [["char"]]
    assert pipe.unloaded == 0


def test_set_adapter_weights_on_module_applies_and_guards():
    module = FakeDenoiserModule()
    spec = LoraSpec(id="c", path="p", weight=0.5, adapter_name="c")
    set_adapter_weights_on_module(module, ("c",), [0.5], adapter_id="t", specs=[spec])
    assert module.set_calls == [(["c"], [0.5])]

    # No module support: a single full-weight adapter is already active (fine),
    # but multiple / non-unity weights cannot be honored.
    set_adapter_weights_on_module(None, ("c",), [1.0], adapter_id="t", specs=[spec])
    with pytest.raises(RuntimeError, match="per-adapter weights"):
        set_adapter_weights_on_module(None, ("c", "d"), [1.0, 0.5], adapter_id="t", specs=[spec])


def test_lora_loader_reuses_overlap_when_adapter_can_delete_targeted_loras(tmp_path):
    first = tmp_path / "style.safetensors"
    second = tmp_path / "detail.safetensors"
    third = tmp_path / "motion.safetensors"
    first.write_bytes(b"lora")
    second.write_bytes(b"lora")
    third.write_bytes(b"lora")
    pipe = FakeTargetedLoraPipe()
    state = apply_loras_to_pipeline(
        pipe,
        [{"id": "style", "installedPath": str(first)}, {"id": "detail", "installedPath": str(second)}],
        adapter_id="diffusers_test",
    )

    apply_loras_to_pipeline(
        pipe,
        [{"id": "style", "installedPath": str(first)}, {"id": "motion", "installedPath": str(third)}],
        adapter_id="diffusers_test",
        previous_state=state,
    )

    assert pipe.unloaded == 0
    assert pipe.deleted == [[state.specs[1].adapter_name]]
    assert [path for path, _name in pipe.loaded] == [str(first), str(second), str(third)]


def test_lora_cache_key_is_stable_for_reordered_loras(tmp_path):
    first = tmp_path / "style.safetensors"
    second = tmp_path / "detail.safetensors"
    first.write_bytes(b"lora")
    second.write_bytes(b"lora")
    left = [{"id": "style", "installedPath": str(first), "weight": 0.5}, {"id": "detail", "installedPath": str(second)}]
    right = list(reversed(left))

    key = lora_cache_key(left)
    assert key == lora_cache_key(right)
    assert len(key) == 64


def test_first_safetensors_path_prefers_final_over_step_checkpoints(tmp_path):
    # A trained-LoRA directory holds the final adapter plus per-step checkpoints.
    for step in (250, 500, 3000):
        (tmp_path / f"kelsie_lora-step{step:06d}.safetensors").write_bytes(b"ckpt")
    final = tmp_path / "kelsie_lora.safetensors"
    final.write_bytes(b"final")

    assert first_safetensors_path(tmp_path) == final


def test_first_safetensors_path_picks_latest_checkpoint_when_no_final(tmp_path):
    for step in (250, 500, 3000):
        (tmp_path / f"kelsie_lora-step{step:06d}.safetensors").write_bytes(b"ckpt")

    assert first_safetensors_path(tmp_path) == tmp_path / "kelsie_lora-step003000.safetensors"


def test_resolve_lora_file_uses_declared_files_over_checkpoints(tmp_path):
    (tmp_path / "kelsie_lora-step000250.safetensors").write_bytes(b"ckpt")
    final = tmp_path / "kelsie_lora.safetensors"
    final.write_bytes(b"final")

    resolved = resolve_lora_file(tmp_path, {"files": ["kelsie_lora.safetensors"]})

    assert resolved == final


def test_lora_loader_allows_single_implicit_weight_without_set_adapters(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")
    pipe = FakeSingleLoraPipe()

    state = apply_loras_to_pipeline(pipe, [{"id": "style", "installedPath": str(first), "weight": 1.0}], adapter_id="diffusers_test")

    assert state.adapter_names
    assert pipe.loaded[0][0] == str(first)


def test_lora_loader_fails_when_pipeline_cannot_load_loras(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")

    with pytest.raises(RuntimeError, match="does not support loading LoRA weights"):
        apply_loras_to_pipeline(object(), [{"id": "style", "installedPath": str(first)}], adapter_id="diffusers_test")


def test_lora_loader_explains_missing_peft_backend(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")

    with pytest.raises(RuntimeError, match="LoRA style requires the PEFT backend") as info:
        apply_loras_to_pipeline(
            FakePeftBackendErrorPipe(),
            [{"id": "style", "installedPath": str(first)}],
            adapter_id="diffusers_test",
        )

    assert isinstance(info.value.__cause__, ValueError)
    assert "docker compose build worker --no-cache" in str(info.value)


def test_lora_loader_detects_reworded_peft_backend_errors(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")

    class MissingPeftPipe:
        def load_lora_weights(self, path, adapter_name=None):
            raise ModuleNotFoundError("No module named 'peft'")

    with pytest.raises(RuntimeError, match="LoRA style requires the PEFT backend"):
        apply_loras_to_pipeline(
            MissingPeftPipe(),
            [{"id": "style", "installedPath": str(first)}],
            adapter_id="diffusers_test",
        )


def test_unsupported_adapter_guard_rejects_loras(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")

    with pytest.raises(RuntimeError, match="does not support LoRA application"):
        reject_loras_if_unsupported([{"id": "style", "installedPath": str(first)}], "procedural_preview")


def test_lora_compatibility_guard_rejects_mismatched_family_before_load(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")
    pipe = FakeLoraPipe()

    with pytest.raises(RuntimeError, match="LoRA style is not compatible with model family z-image"):
        apply_loras_to_pipeline(
            pipe,
            [{"id": "style", "installedPath": str(first), "family": "qwen_image"}],
            adapter_id="diffusers_test",
            model_family="z-image",
        )

    assert pipe.loaded == []


def test_lora_compatibility_guard_soft_passes_legacy_jobs_without_family(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")
    pipe = FakeLoraPipe()

    apply_loras_to_pipeline(
        pipe,
        [{"id": "style", "installedPath": str(first)}],
        adapter_id="diffusers_test",
        model_family="z-image",
    )

    assert pipe.loaded[0][0] == str(first)


def test_lora_compatibility_guard_accepts_normalized_family_aliases():
    validate_lora_compatibility(
        [{"id": "style", "compatibility": {"families": ["z_image"]}}],
        model_family="z-image",
        adapter_id="diffusers_test",
    )


def test_lora_compatibility_chroma_accepts_flux_but_not_vice_versa():
    # Chroma accepts flux LoRAs (FLUX.1-schnell-derived, identical keys). sc-1832.
    validate_lora_compatibility(
        [{"id": "flux_style", "compatibility": {"families": ["flux"]}}],
        model_family="chroma",
        adapter_id="diffusers_test",
    )
    # ...and chroma-tagged LoRAs.
    validate_lora_compatibility(
        [{"id": "chroma_style", "compatibility": {"families": ["chroma"]}}],
        model_family="chroma",
        adapter_id="diffusers_test",
    )
    # The relationship is one-directional: a Flux model rejects a chroma LoRA.
    with pytest.raises(RuntimeError, match="not compatible with model family flux"):
        validate_lora_compatibility(
            [{"id": "chroma_style", "compatibility": {"families": ["chroma"]}}],
            model_family="flux",
            adapter_id="diffusers_test",
        )


def test_lora_compatibility_flux2_klein_accepts_flux2_lora():
    # FLUX.2 [klein] models have family "flux2-klein" but accept "flux2" LoRAs
    # (loraCompatibility.families = ["flux2"]). The validate guard keys off the
    # model `family` string, so the klein->flux2 relationship must be declared.
    validate_lora_compatibility(
        [{"id": "portrait_engine", "compatibility": {"families": ["flux2"]}}],
        model_family="flux2-klein",
        adapter_id="mlx_flux2",
    )
    # The relationship is one-directional: a plain flux2 model would reject a
    # flux2-klein-tagged LoRA (no such model ships today, but keep the gate tight).
    with pytest.raises(RuntimeError, match="not compatible with model family flux2"):
        validate_lora_compatibility(
            [{"id": "klein_only", "compatibility": {"families": ["flux2-klein"]}}],
            model_family="flux2",
            adapter_id="mlx_flux2",
        )


def test_lora_weight_defaults_on_unparseable_values():
    assert lora_weight({"weight": "not-a-number"}) == 0.8


def test_lora_specs_fail_before_inference_for_missing_or_excess_loras(tmp_path):
    missing = tmp_path / "missing.safetensors"

    with pytest.raises(RuntimeError, match="file is missing"):
        normalize_lora_specs([{"id": "missing", "installedPath": str(missing)}])

    empty_dir = tmp_path / "empty_lora"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match=r"LoRA empty has no \.safetensors file"):
        normalize_lora_specs([{"id": "empty", "installedPath": str(empty_dir)}])

    many = [{"id": f"lora_{index}", "installedPath": str(missing)} for index in range(4)]
    with pytest.raises(RuntimeError, match="at most 3 LoRAs"):
        normalize_lora_specs(many)


def test_lora_specs_resolve_installed_directory_to_safetensors_file(tmp_path):
    # Installed LoRAs are stored as a directory and the Rust API reports the
    # directory as `installedPath`. The native ltx-core loader mmaps the path
    # directly, and mmap on a directory raises ENODEV ("No such device (os error
    # 19)"), so the spec path must point at the .safetensors file, not the dir.
    lora_dir = tmp_path / "loras" / "lauren"
    lora_dir.mkdir(parents=True)
    weights = lora_dir / "Lauren_ltx2.3.safetensors"
    weights.write_bytes(b"")

    specs = normalize_lora_specs(
        [
            {
                "id": "lauren",
                "weight": 0.8,
                "files": ["Lauren_ltx2.3.safetensors"],
                "installedPath": str(lora_dir),
                "source": {"provider": "local", "path": "loras/lauren"},
            }
        ]
    )

    assert specs[0].path == str(weights)


def test_lora_specs_resolve_huggingface_cache_snapshot(monkeypatch, tmp_path):
    cache_root = tmp_path / "hf" / "hub"
    snapshot = write_huggingface_cache_resource(
        cache_root,
        "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    )
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))

    specs = normalize_lora_specs(
        [
            {
                "id": "ltx_2_3_ic_union_control",
                "weight": 0.7,
                "source": {
                    "provider": "huggingface",
                    "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                    "file": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                },
            }
        ]
    )

    assert specs[0].path == str(snapshot / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")
    assert specs[0].weight == 0.7


def test_lora_specs_prefer_huggingface_ref_main_snapshot(monkeypatch, tmp_path):
    cache_root = tmp_path / "hf" / "hub"
    repo = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
    file_name = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    write_huggingface_cache_resource(cache_root, repo, file_name, revision="aaa111")
    main_snapshot = write_huggingface_cache_resource(cache_root, repo, file_name, revision="zzz999", refs_main=True)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))

    specs = normalize_lora_specs(
        [
            {
                "id": "ltx_2_3_ic_union_control",
                "source": {
                    "provider": "huggingface",
                    "repo": repo,
                    "file": file_name,
                },
            }
        ]
    )

    assert specs[0].path == str(main_snapshot / file_name)


def test_image_adapter_env_aliases_and_unknown_values(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "procedural")
    assert create_image_adapter({"payload": {"model": "z_image_turbo"}}).__class__.__name__ == "ProceduralImageAdapter"

    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "typo")
    try:
        create_image_adapter({"payload": {"model": "z_image_turbo"}})
    except RuntimeError as exc:
        assert "Unsupported SCENEWORKS_IMAGE_ADAPTER" in str(exc)
    else:
        raise AssertionError("Unknown image adapter override should fail loudly.")


def test_create_image_adapter_routes_lens_turbo():
    adapter = create_image_adapter({"payload": {"model": "lens_turbo"}})
    assert adapter.__class__.__name__ == "LensTurboAdapter"
    assert adapter.id == "lens_turbo"


def test_image_adapter_env_override_selects_lens(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "lens_turbo")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "LensTurboAdapter"


def test_lens_turbo_model_target_defaults():
    target = MODEL_TARGETS["lens_turbo"]
    assert target["adapter"] == "lens_turbo"
    assert target["family"] == "lens"
    assert target["steps"] == 4
    assert target["supportsEdit"] is False
    assert target["repo"] == "microsoft/Lens-Turbo"


def test_lens_base_model_target_defaults():
    target = MODEL_TARGETS["lens"]
    assert target["adapter"] == "lens_turbo"
    assert target["family"] == "lens"
    # Non-distilled base: 20-step / CFG 5.0 (vs Turbo's 4 / 1.0).
    assert target["steps"] == 20
    assert target["guidanceScale"] == 5.0
    assert target["repo"] == "microsoft/Lens"


def test_lens_guidance_scale_uses_per_model_default_and_override():
    adapter = LensTurboAdapter()
    base = MODEL_TARGETS["lens"]
    turbo = MODEL_TARGETS["lens_turbo"]
    # The per-model default applies when the request does not override guidance.
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), base) == 5.0
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), turbo) == 1.0
    # An explicit request value wins for either variant.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 3.5}), base) == 3.5


def test_lens_turbo_rejects_image_edit(tmp_path):
    job = {
        "id": "job_lens_edit",
        "payload": {
            "projectId": "project_x",
            "mode": "edit_image",
            "model": "lens_turbo",
            "prompt": "a cat",
        },
    }
    noop = lambda *args, **kwargs: None  # noqa: E731
    try:
        LensTurboAdapter().generate(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )
    except RuntimeError as exc:
        assert "does not support image editing" in str(exc)
    else:
        raise AssertionError("Lens-Turbo is text-to-image only and must reject edit_image.")


def test_lens_turbo_requires_sidecar_when_missing(monkeypatch):
    # Point the sidecar interpreter at a path that does not exist so the adapter
    # reports the actionable "rebuild with INCLUDE_LENS" error instead of trying
    # to import the (main-venv-incompatible) lens stack in-process.
    monkeypatch.setenv("SCENEWORKS_LENS_PYTHON", "/nonexistent/lens-venv/bin/python")
    job = {
        "id": "job_lens_t2i",
        "payload": {
            "projectId": "project_x",
            "mode": "text_to_image",
            "model": "lens_turbo",
            "prompt": "a cat",
        },
    }
    noop = lambda *args, **kwargs: None  # noqa: E731
    try:
        LensTurboAdapter().generate(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )
    except RuntimeError as exc:
        assert "sidecar" in str(exc).lower()
    else:
        raise AssertionError("Lens generation must fail clearly when the sidecar venv is unavailable.")


def test_lens_train_runner_reads_network_type_and_factor():
    from scene_worker import lens_train_runner as lt

    assert lt._network_type({}) == "lora"
    assert lt._decompose_factor({}) == -1
    assert lt._network_type({"advanced": {"networkType": "LoKr"}}) == "lokr"
    assert lt._decompose_factor({"advanced": {"decomposeFactor": 8}}) == 8
    # Bad factor falls back to auto (-1).
    assert lt._decompose_factor({"advanced": {"decomposeFactor": "x"}}) == -1
    # Default targets point at the Linear `to_out.0`, not the `to_out` ModuleList
    # (PEFT errors on the ModuleList — sc-2218).
    assert lt._target_modules({}) == ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"]
    assert "to_out" not in lt._target_modules({})  # the bare ModuleList name is gone


def test_lens_train_runner_build_network_config():
    from scene_worker import lens_train_runner as lt

    fake_peft = SimpleNamespace(
        LoraConfig=lambda **kw: ("lora", kw),
        LoKrConfig=lambda **kw: ("lokr", kw),
    )
    kind, _ = lt._build_network_config(
        fake_peft, network_type="lora", rank=8, alpha=8, decompose_factor=-1,
        target_modules=["img_qkv", "txt_qkv"],
    )
    assert kind == "lora"

    kind, kwargs = lt._build_network_config(
        fake_peft, network_type="lokr", rank=8, alpha=16, decompose_factor=4,
        target_modules=["img_qkv", "txt_qkv", "to_out.0", "to_add_out"],
    )
    assert kind == "lokr"
    assert kwargs["r"] == 8 and kwargs["alpha"] == 16
    assert kwargs["decompose_factor"] == 4
    assert kwargs["target_modules"] == ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"]


def test_lens_save_lora_routes_by_network_type(monkeypatch, tmp_path):
    from scene_worker import lens_train_runner as lt

    # Plain LoRA: delegates to the transformer's diffusers PeftAdapterMixin saver.
    saved: dict[str, object] = {}

    class FakeTransformer:
        def save_lora_adapter(self, output_dir, *, weight_name=None, safe_serialization=None):
            saved["output_dir"] = output_dir
            saved["weight_name"] = weight_name

    out = lt._save_lora(FakeTransformer(), str(tmp_path), "lens.safetensors", network_type="lora")
    assert saved["weight_name"] == "lens.safetensors"
    assert out == str(tmp_path / "lens.safetensors")

    # LoKr: routes to the raw metadata writer instead (lokr_w1/w2 aren't
    # save_lora_adapter-compatible), and never calls save_lora_adapter.
    captured: dict[str, object] = {}

    def fake_save_lokr(transformer, output_dir, file_name, **kwargs):
        captured["file_name"] = file_name
        captured["kwargs"] = kwargs
        return str(tmp_path / file_name)

    monkeypatch.setattr(lt, "_save_lokr_adapter", fake_save_lokr)

    class ExplodingTransformer:
        def save_lora_adapter(self, *a, **k):
            raise AssertionError("LoKr must not save via save_lora_adapter")

    out = lt._save_lora(
        ExplodingTransformer(), str(tmp_path), "lens.safetensors",
        network_type="lokr", rank=8, alpha=8, decompose_factor=4,
        target_modules=["img_qkv", "txt_qkv"],
    )
    assert captured["file_name"] == "lens.safetensors"
    assert captured["kwargs"] == {
        "rank": 8, "alpha": 8, "decompose_factor": 4,
        "target_modules": ["img_qkv", "txt_qkv"],
    }


def test_lens_save_lokr_adapter_stamps_metadata(monkeypatch, tmp_path):
    import sys
    import types as types_module

    from scene_worker import lens_train_runner as lt

    fake_state = {"transformer.x.lokr_w1": "t1", "transformer.x.lokr_w2": "t2"}
    fake_peft_utils = types_module.ModuleType("peft.utils")
    fake_peft_utils.get_peft_model_state_dict = lambda _m: fake_state
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)

    written: dict[str, object] = {}

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

    fake_st_torch = types_module.ModuleType("safetensors.torch")

    def fake_save_file(tensors, path, metadata=None):
        written["tensors"] = dict(tensors)
        written["metadata"] = metadata

    fake_st_torch.save_file = fake_save_file
    monkeypatch.setitem(sys.modules, "safetensors", types_module.ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    # get_peft_model_state_dict returns our fake tensors; patch their .detach chain.
    monkeypatch.setattr(fake_peft_utils, "get_peft_model_state_dict", lambda _m: {
        "transformer.x.lokr_w1": FakeTensor(), "transformer.x.lokr_w2": FakeTensor(),
    })

    out = lt._save_lokr_adapter(
        object(), str(tmp_path), "lens_lokr.safetensors",
        rank=8, alpha=16, decompose_factor=4, target_modules=["img_qkv", "txt_qkv"],
    )
    assert out == str(tmp_path / "lens_lokr.safetensors")
    meta = written["metadata"]
    assert meta["networkType"] == "lokr"
    assert meta["rank"] == "8" and meta["alpha"] == "16" and meta["decomposeFactor"] == "4"
    assert json.loads(meta["targetModules"]) == ["img_qkv", "txt_qkv"]
    assert set(written["tensors"]) == {"transformer.x.lokr_w1", "transformer.x.lokr_w2"}


def test_lens_runner_apply_loras_routes_lokr_to_injection(monkeypatch):
    from scene_worker import lens_runner as lr

    # Classify by recorded networkType (stubbed so the test needs no real files).
    kinds = {"/lora.safetensors": "lora", "/lokr.safetensors": "lokr"}
    monkeypatch.setattr(lr, "_adapter_network_type", lambda path: kinds[path])

    injected: list[str] = []
    monkeypatch.setattr(lr, "_inject_lokr", lambda t, path, name: injected.append(name))

    loaded: list[str] = []
    set_calls: dict[str, object] = {}

    class FakeTransformer:
        def __init__(self):
            self.peft_config: dict = {}

        def load_lora_adapter(self, path, adapter_name=None, prefix="__unset__"):
            loaded.append(adapter_name)
            # Real PeftAdapterMixin registers the adapter once a prefix matches
            # keys; _load_plain_lora confirms registration before moving on.
            self.peft_config[adapter_name] = object()

        def set_adapters(self, names, weights=None):
            set_calls["names"] = names
            set_calls["weights"] = weights

    lr._apply_loras(
        FakeTransformer(),
        [
            {"path": "/lora.safetensors", "weight": 0.8, "name": "a"},
            {"path": "/lokr.safetensors", "weight": 1.0, "name": "b"},
        ],
    )
    # LoKr → PEFT injection; plain LoRA → load_lora_adapter; both then scaled together.
    assert injected == ["b"]
    assert loaded == ["a"]
    assert set_calls["names"] == ["a", "b"]
    assert set_calls["weights"] == [0.8, 1.0]


def test_lens_runner_loads_plain_lora_with_bare_key_prefix(monkeypatch):
    """The Lens trainer (``save_lora_adapter``) writes *bare* module keys, but
    ``load_lora_adapter`` defaults to a ``prefix='transformer'`` filter that
    matches zero of them and only *warns* — so a naive single call silently loads
    nothing and generation falls back to the base model. ``_load_plain_lora`` must
    try ``prefix=None`` (the trainer's layout) and confirm the adapter actually
    registered (sc-2218; surfaced by the first real-HW plain-LoRA round-trip)."""
    from scene_worker import lens_runner as lr

    monkeypatch.setattr(lr, "_adapter_network_type", lambda path: "lora")

    class FakeTransformer:
        def __init__(self):
            self.peft_config: dict = {}
            self.load_calls: list = []
            self.set_calls: dict = {}

        def load_lora_adapter(self, path, adapter_name=None, prefix="__unset__"):
            self.load_calls.append(prefix)
            if prefix is None:  # bare-key layout the Lens trainer writes
                self.peft_config[adapter_name] = object()

        def set_adapters(self, names, weights=None):
            self.set_calls = {"names": names, "weights": weights}

    ft = FakeTransformer()
    lr._apply_loras(ft, [{"path": "/lora.safetensors", "weight": 0.7, "name": "a"}])
    assert ft.load_calls == [None]  # one call, prefix=None — no silent prefix miss
    assert "a" in ft.peft_config  # adapter actually registered
    assert ft.set_calls == {"names": ["a"], "weights": [0.7]}


def test_lens_runner_plain_lora_falls_back_to_transformer_prefix(monkeypatch):
    """An externally-saved adapter keyed under ``transformer.`` still loads: the
    ``prefix=None`` attempt registers nothing, so ``_load_plain_lora`` retries with
    the ``transformer`` prefix before giving up."""
    from scene_worker import lens_runner as lr

    monkeypatch.setattr(lr, "_adapter_network_type", lambda path: "lora")

    class FakeTransformer:
        def __init__(self):
            self.peft_config: dict = {}
            self.load_calls: list = []

        def load_lora_adapter(self, path, adapter_name=None, prefix="__unset__"):
            self.load_calls.append(prefix)
            if prefix == "transformer":  # only the prefixed layout matches
                self.peft_config[adapter_name] = object()

        def set_adapters(self, names, weights=None):
            pass

    ft = FakeTransformer()
    lr._apply_loras(ft, [{"path": "/lora.safetensors", "weight": 1.0, "name": "a"}])
    assert ft.load_calls == [None, "transformer"]
    assert "a" in ft.peft_config


def test_lens_runner_plain_lora_raises_when_no_prefix_matches(monkeypatch):
    """If neither key layout registers the adapter, fail loudly instead of
    silently generating from the base model (the pre-fix failure mode)."""
    from scene_worker import lens_runner as lr

    monkeypatch.setattr(lr, "_adapter_network_type", lambda path: "lora")

    class FakeTransformer:
        peft_config: dict = {}

        def load_lora_adapter(self, path, adapter_name=None, prefix="__unset__"):
            pass  # never registers under any prefix

        def set_adapters(self, names, weights=None):
            pass

    with pytest.raises(ValueError, match="matched no transformer modules"):
        lr._apply_loras(
            FakeTransformer(), [{"path": "/lora.safetensors", "name": "a"}]
        )


def test_lens_resolution_for_snaps_to_buckets():
    # Square requests pick the base by area: <1024*1440 px -> 1024, else 1440.
    assert lens_resolution_for(1024, 1024) == (1024, "1:1")
    assert lens_resolution_for(1440, 1440) == (1440, "1:1")
    assert lens_resolution_for(2048, 2048) == (1440, "1:1")
    # Aspect ratio snaps by closest log-ratio (W:H).
    assert lens_resolution_for(1280, 720) == (1024, "16:9")
    assert lens_resolution_for(720, 1280) == (1024, "9:16")
    assert lens_resolution_for(1152, 864) == (1024, "4:3")
    assert lens_resolution_for(864, 1152) == (1024, "3:4")


def test_create_image_adapter_routes_flux_schnell_and_dev():
    # Epic 3018 cutover (sc-3032): FLUX.1 on Mac is claimed by the Rust `mlx` GPU
    # worker; the Python worker always routes the torch FluxDiffusersAdapter.
    schnell = create_image_adapter({"payload": {"model": "flux_schnell"}})
    assert schnell.__class__.__name__ == "FluxDiffusersAdapter"
    assert schnell.id == "flux_diffusers"
    dev = create_image_adapter({"payload": {"model": "flux_dev"}})
    assert dev.__class__.__name__ == "FluxDiffusersAdapter"


def test_image_adapter_env_override_selects_flux(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "flux_diffusers")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "FluxDiffusersAdapter"


def _manifest_brace_walker():
    # Helper for the mlx-block manifest tests. Returns (raw, find_entry_block,
    # find_mlx_block) that walk balanced braces so a URL containing `//` (in
    # the entry text) doesn't trip a naive jsonc strip.
    from pathlib import Path

    manifest_path = Path(__file__).resolve().parent.parent / "config" / "manifests" / "builtin.models.jsonc"
    raw = manifest_path.read_text(encoding="utf-8")

    def find_balanced_block(start_index: int) -> str:
        depth = 0
        for index in range(start_index, len(raw)):
            ch = raw[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start_index : index + 1]
        raise AssertionError(f"unterminated brace block from index {start_index}")

    def find_entry_block(model_id: str) -> str:
        anchor = raw.index(f'"id": "{model_id}"')
        start = raw.rfind("{", 0, anchor)
        assert start != -1, f"entry start brace for {model_id} not found"
        return find_balanced_block(start)

    def find_mlx_block(entry_block: str) -> str:
        import re

        match = re.search(r'"mlx"\s*:\s*\{', entry_block)
        assert match, "entry block has no mlx block"
        # Resolve the entry block's position in the raw manifest, then walk
        # balanced braces from the actual opening brace so nested limits {...}
        # are captured (Qwen carries a sampler/scheduler limits override, FLUX
        # does not).
        entry_start = raw.index(entry_block)
        mlx_open = entry_start + match.end() - 1
        return find_balanced_block(mlx_open)

    return raw, find_entry_block, find_mlx_block


def test_flux_manifest_has_mlx_block():
    # Manifest-driven auto-dispatch + Model Manager memory tier (sc-1970).
    # The Rust API owns the canonical jsonc parser; here we just confirm both
    # FLUX entries carry an `mlx` block and the contents look right.
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()

    for model_id in ("flux_schnell", "flux_dev"):
        block = find_entry_block(model_id)
        mlx_block = find_mlx_block(block)
        quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
        mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
        assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
            f"{model_id} mlx.quantize must be a supported quant level (sc-1970)"
        )
        assert mem_match and int(mem_match.group(1)) > 0, (
            f"{model_id} mlx.minMemoryGb must be a positive int (sc-1970)"
        )


def test_qwen_image_manifest_has_mlx_block():
    # sc-1972: qwen_image carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2
    # precedent of restricting the menu to default-only when the MLX path is
    # the active backend, epic 1753 §14).
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("qwen_image")
    mlx_block = find_mlx_block(block)
    quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
        "qwen_image mlx.quantize must be a supported quant level (sc-1972)"
    )
    assert mem_match and int(mem_match.group(1)) > 0, (
        "qwen_image mlx.minMemoryGb must be a positive int (sc-1972)"
    )
    # MLX sampler/scheduler menu override
    assert '"samplers": ["default"]' in mlx_block, (
        "qwen_image mlx must restrict samplers to default (mflux loop is linear-only)"
    )
    assert '"schedulers": ["default"]' in mlx_block, (
        "qwen_image mlx must restrict schedulers to default (mflux loop is linear-only)"
    )


def test_create_image_adapter_routes_qwen_image():
    # Epic 3018 cutover (sc-3032): Qwen-Image on Mac is claimed by the Rust `mlx`
    # GPU worker; the Python worker always routes the torch QwenImageAdapter.
    adapter = create_image_adapter({"payload": {"model": "qwen_image"}})
    assert adapter.__class__.__name__ == "QwenImageAdapter"
    assert adapter.id == "qwen_image"


def test_flux2_true_v2_manifest_install_time_conversion():
    # sc-2235: the entry must declare the install-time conversion contract the
    # Rust convert job + adapter rely on.
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("flux2_klein_9b_true_v2")
    assert '"macOnly": true' in block
    assert '"adapter": "mlx_flux2"' in block
    # Only the bf16 single-file is pulled (not the whole 73 GB repo).
    assert "Flux2-Klein-9B-True-v2-bf16.safetensors" in block
    # Undistilled defaults differ from the 4-step distill.
    assert re.search(r'"steps"\s*:\s*24', block)

    mlx_block = find_mlx_block(block)
    assert '"requiresConversion": true' in mlx_block
    assert '"converter": "flux2_klein_diffusers"' in mlx_block
    assert '"convertSourceRepo": "wikeeyang/Flux2-Klein-9B-True-V2"' in mlx_block
    assert '"convertBaseRepo": "black-forest-labs/FLUX.2-klein-9B"' in mlx_block
    assert re.search(r'"quantize"\s*:\s*8', mlx_block)


def test_runtime_registry_covers_all_model_target_adapters():
    # sc-2203 guard: every adapter id a manifest model can dispatch to (the
    # `adapter` field in MODEL_TARGETS) must be a key in the runtime's
    # image_adapters registry, or create_image_adapter's `adapters.get(id)`
    # returns None and the job crashes. Catches "added to dispatch but forgot
    # to register" for any future adapter.
    #
    # Epic 3018 cutover (sc-3032): the MLX adapter ids are the exception — the
    # Python MLX adapters were deleted, so MLX-eligible models are claimed by the
    # Rust `mlx` GPU worker and never dispatch to a Python adapter (FLUX.2-klein,
    # the only MLX-only family, makes create_image_adapter raise). They must NOT
    # be registered on the Python worker.
    import re
    from pathlib import Path

    from scene_worker import runtime
    from scene_worker.image_adapters import MODEL_TARGETS

    # MLX adapter ids the Python worker no longer owns (Rust `mlx` worker claims them).
    rust_mlx_only = {"mlx_flux", "mlx_qwen", "mlx_z_image", "mlx_flux2"}

    src = Path(runtime.__file__).read_text(encoding="utf-8")
    block = src.split("image_adapters: dict[str, object] = {", 1)[1].split("}", 1)[0]
    registered = set(re.findall(r'"([a-z0-9_]+)":', block))

    needed = {
        target["adapter"]
        for target in MODEL_TARGETS.values()
        if target.get("adapter") and target["adapter"] not in rust_mlx_only
    }
    missing = needed - registered
    assert not missing, f"adapter ids in MODEL_TARGETS not registered in runtime: {sorted(missing)}"
    # The MLX adapters are intentionally absent from the Python registry post-cutover.
    assert not (registered & rust_mlx_only)


def test_request_has_lokr_lora_detection():
    from scene_worker.image_adapters import _request_has_lokr_lora

    # Recorded networkType (top-level, mirroring baseModel) or nested in
    # compatibility — both are read with no file I/O (epic 2193).
    assert _request_has_lokr_lora({"loras": [{"id": "a", "networkType": "lokr"}]}) is True
    assert _request_has_lokr_lora({"loras": [{"id": "a", "compatibility": {"networkType": "LoKr"}}]}) is True
    assert _request_has_lokr_lora({"loras": [{"id": "a", "networkType": "lora"}]}) is False
    assert _request_has_lokr_lora({"loras": []}) is False
    assert _request_has_lokr_lora({}) is False


def test_flux2_klein_manifest_entries_present():
    # Both flux2_klein_9b and flux2_klein_9b_kv must be present in the
    # builtin manifest with the expected adapter + family + mlx block.
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    # Both ids expose the same capability set: -kv is no longer gated to
    # character_image only — it runs plain txt2img on par with the base 9B,
    # the cache just doesn't engage without a reference (sc-2173).
    for model_id in ("flux2_klein_9b", "flux2_klein_9b_kv"):
        block = find_entry_block(model_id)
        assert '"adapter": "mlx_flux2"' in block, model_id
        assert '"family": "flux2-klein"' in block, model_id
        assert '"macOnly": true' in block, model_id
        assert '"gated": true' in block, model_id
        mlx_block = find_mlx_block(block)
        quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
        assert quant_match is not None, f"{model_id}: mlx.quantize missing"
        assert int(quant_match.group(1)) == 8, f"{model_id}: quantize should be 8 (sweet spot)"
        assert '"text_to_image"' in block, model_id
        assert '"character_image"' in block, model_id


# --- sc-2145: Z-Image MLX adapter ---


def test_z_image_turbo_manifest_has_mlx_block():
    # sc-2145: z_image_turbo carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2 /
    # qwen_image precedents of restricting the menu to default-only when the
    # MLX path is the active backend, epic 1753 §14).
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("z_image_turbo")
    mlx_block = find_mlx_block(block)
    quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
        "z_image_turbo mlx.quantize must be a supported quant level (sc-2145)"
    )
    assert mem_match and int(mem_match.group(1)) > 0, (
        "z_image_turbo mlx.minMemoryGb must be a positive int (sc-2145)"
    )
    assert '"samplers": ["default"]' in mlx_block, (
        "z_image_turbo mlx must restrict samplers to default (mflux loop is linear-only)"
    )
    assert '"schedulers": ["default"]' in mlx_block, (
        "z_image_turbo mlx must restrict schedulers to default (mflux loop is linear-only)"
    )


def test_create_image_adapter_routes_z_image_turbo():
    # Epic 3018 cutover (sc-3032): Z-Image on Mac is claimed by the Rust `mlx` GPU
    # worker; the Python worker always routes the torch ZImageDiffusersAdapter.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "ZImageDiffusersAdapter"
    assert adapter.id == "z_image_diffusers"


def test_sdxl_manifest_has_mlx_block():
    # sc-1975: sdxl carries an mlx block (no `limits` override here — Apple's
    # SDXL schedule already matches the torch EulerDiscrete default, and
    # there's no per-model sampler menu in the sdxl manifest entry to limit).
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("sdxl")
    mlx_block = find_mlx_block(block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert mem_match and int(mem_match.group(1)) > 0, (
        "sdxl mlx.minMemoryGb must be a positive int (sc-1975)"
    )
    # No quantize key in v1 — Apple's Q8 recipe breaks SDXL base 1.0 (see
    # sc-1975 spike finding). bf16 is the only supported precision.
    assert '"quantize"' not in mlx_block, (
        "sdxl mlx block must not declare a quantize default in v1 (sc-1975: "
        "Apple's Q8 recipe breaks on SDXL base 1.0; defer until calibration lands)"
    )


def test_sdxl_auto_dispatch_uses_torch_adapter_on_python_worker(monkeypatch):
    # sc-3060 retired the in-process vendored MLX SDXL adapter (sc-1975). On the Python
    # worker, SDXL now always resolves to the torch SdxlDiffusersAdapter — on every host
    # and for every shape (txt2img / edit / reference). MLX routing for SDXL lives in the
    # Rust API claim layer (jobs_store::sdxl_mlx_eligible), which defers eligible jobs to
    # the `mlx` GPU worker; those never reach this Python adapter.
    monkeypatch.delenv("SCENEWORKS_IMAGE_ADAPTER", raising=False)
    for platform in ("darwin", "linux", "win32"):
        monkeypatch.setattr("sys.platform", platform)
        for payload in (
            {"model": "sdxl"},
            {"model": "sdxl", "mode": "edit_image"},
            {"model": "sdxl", "referenceAssetId": "asset_ref"},
            {"model": "realvisxl"},
        ):
            adapter = create_image_adapter({"payload": payload})
            assert adapter.__class__.__name__ == "SdxlDiffusersAdapter", (
                f"{payload} on {platform} must use the torch SDXL adapter"
            )


def test_flux_model_target_defaults():
    schnell = MODEL_TARGETS["flux_schnell"]
    assert schnell["adapter"] == "flux_diffusers"
    assert schnell["family"] == "flux"
    assert schnell["supportsEdit"] is False
    # Guidance-distilled: 4 steps, guidance 0, T5 max_seq_len 256.
    assert schnell["steps"] == 4
    assert schnell["guidanceScale"] == 0.0
    assert schnell["maxSequenceLength"] == 256
    assert schnell["repo"] == "black-forest-labs/FLUX.1-schnell"

    dev = MODEL_TARGETS["flux_dev"]
    assert dev["adapter"] == "flux_diffusers"
    assert dev["family"] == "flux"
    assert dev["supportsEdit"] is False
    # Guided: 28 steps, guidance 3.5, T5 max_seq_len 512.
    assert dev["steps"] == 28
    assert dev["guidanceScale"] == 3.5
    assert dev["maxSequenceLength"] == 512
    assert dev["repo"] == "black-forest-labs/FLUX.1-dev"
    # XLabs FLUX IP-Adapter is the diffusers-blessed character-image path; CLIP-L
    # encoder, no subfolder. flux_schnell has no native IP-Adapter trained for it,
    # so the block lives on flux_dev only (sc-2011).
    ip_adapter = dev["ipAdapter"]
    assert ip_adapter["repo"] == "XLabs-AI/flux-ip-adapter"
    assert ip_adapter["weight"] == "ip_adapter.safetensors"
    assert ip_adapter["imageEncoderRepo"] == "openai/clip-vit-large-patch14"
    assert "ipAdapter" not in schnell


def test_flux_guidance_scale_uses_per_model_default_and_override():
    adapter = FluxDiffusersAdapter()
    schnell = MODEL_TARGETS["flux_schnell"]
    dev = MODEL_TARGETS["flux_dev"]
    # Per-model default applies when the request does not override guidance.
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), schnell) == 0.0
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), dev) == 3.5
    # An explicit request value wins.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 2.0}), dev) == 2.0
    # Unparseable override falls back to the per-model default.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": "x"}), dev) == 3.5


def test_flux_num_inference_steps_default_and_override():
    adapter = FluxDiffusersAdapter()
    schnell = MODEL_TARGETS["flux_schnell"]
    dev = MODEL_TARGETS["flux_dev"]
    # Defaults come straight from the model target (no Z-Image +1 quirk).
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), schnell) == 4
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), dev) == 28
    # Explicit override is honored and clamped to [1, 80].
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 12}), schnell) == 12
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), dev) == 80


def test_flux_max_sequence_length_default_and_override():
    adapter = FluxDiffusersAdapter()
    schnell = MODEL_TARGETS["flux_schnell"]
    dev = MODEL_TARGETS["flux_dev"]
    assert adapter._max_sequence_length(SimpleNamespace(advanced={}), schnell) == 256
    assert adapter._max_sequence_length(SimpleNamespace(advanced={}), dev) == 512
    # Override honored, clamped to the T5 max of 512.
    assert adapter._max_sequence_length(SimpleNamespace(advanced={"maxSequenceLength": 128}), dev) == 128
    assert adapter._max_sequence_length(SimpleNamespace(advanced={"maxSequenceLength": 4096}), dev) == 512


def test_flux_reference_asset_id_parsed():
    request = image_request_from_job(
        {"payload": {"projectId": "p", "model": "flux_dev", "referenceAssetId": "asset-ref"}}
    )
    assert request.reference_asset_id == "asset-ref"


def test_flux_use_ip_adapter_only_for_text_with_reference():
    use = FluxDiffusersAdapter._use_ip_adapter
    # IP-Adapter runs on the T2I pipeline with a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id="a")) is True
    # FLUX is T2I-only today but the gate still mirrors SDXL/Kolors so reference +
    # img2img (FLUX.1 Kontext) can opt in cleanly when that path lands.
    assert use(SimpleNamespace(mode="edit_image", reference_asset_id="a")) is False
    # No reference image → no IP-Adapter regardless of mode.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id=None)) is False


def test_flux_ip_adapter_scale_default_and_clamp():
    adapter = FluxDiffusersAdapter()
    # XLabs+CLIP-L is the resemblance tier (faithful identity = PuLID-FLUX); the
    # default 0.7 matches SDXL plus-face — same headroom for the prompt.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={})) == 0.7
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 0.4})) == 0.4
    # Clamped to [0, 1]; unparseable falls back to the default.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 5})) == 1.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": -2})) == 0.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": "x"})) == 0.7


def test_flux_true_cfg_scale_default_and_clamp():
    adapter = FluxDiffusersAdapter()
    # FLUX is guidance-distilled, so real CFG against negative_prompt rides on
    # the parallel true_cfg_scale kwarg. XLabs docs default 4.0; clamp [1, 10]
    # (below 1.0 disables CFG, above 10 hard-bakes the negative prompt).
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={})) == 4.0
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 2.5})) == 2.5
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 0.0})) == 1.0
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 99})) == 10.0
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": "x"})) == 4.0


def test_flux_reference_run_pipeline_passes_ip_adapter_image_and_true_cfg(tmp_path, monkeypatch):
    """A T2I FLUX job with referenceAssetId drives the IP-Adapter branch of
    _run_pipeline: load_reference_image(project_path, reference_asset_id) →
    ip_adapter_image kwarg, set_ip_adapter_scale() per request, and true_cfg_scale
    + negative_prompt onto kwargs. Mirrors the SDXL torch-free pattern."""

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.scales: list[float] = []
            self.last_kwargs: dict[str, Any] = {}

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        # Named params so filter_call_kwargs keeps the FLUX-specific kwargs we
        # need to assert (it introspects __call__ via inspect.signature and drops
        # anything not in the accepted-name set; a bare **kwargs is treated as
        # accepting nothing because the var-keyword param itself is the only
        # name in the signature).
        def __call__(
            self,
            *,
            prompt=None,
            negative_prompt=None,
            ip_adapter_image=None,
            true_cfg_scale=None,
            height=None,
            width=None,
            num_inference_steps=None,
            guidance_scale=None,
            max_sequence_length=None,
            generator=None,
            **kwargs,
        ):
            self.last_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "ip_adapter_image": ip_adapter_image,
                "true_cfg_scale": true_cfg_scale,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_sequence_length,
                "generator": generator,
                **kwargs,
            }
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen: list[tuple] = []

    def fake_load_reference_image(project_path, reference_asset_id):
        seen.append((project_path, reference_asset_id))
        return FakeImage()

    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image", fake_load_reference_image
    )

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_image",
                "model": "flux_dev",
                "prompt": "a portrait of the character",
                "negativePrompt": "blurry",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {"ipAdapterScale": 0.5, "trueCfgScale": 3.0},
            }
        }
    )
    pipe = FakePipe()
    result = FluxDiffusersAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"), pipe, request, 7, project_path
    )
    # IP-Adapter branch ran: reference loaded → ip_adapter_image kwarg, per-request
    # scale applied, and true_cfg_scale + negative_prompt threaded through.
    assert seen == [(project_path, "asset-ref")]
    assert pipe.scales == [0.5]
    assert pipe.last_kwargs["true_cfg_scale"] == 3.0
    assert pipe.last_kwargs["negative_prompt"] == "blurry"
    assert pipe.last_kwargs["ip_adapter_image"] is not None
    assert result is FakeOutput.images[0]


def test_flux_rejects_image_edit():
    job = {
        "id": "job_flux_edit",
        "payload": {
            "projectId": "project_x",
            "mode": "edit_image",
            "model": "flux_dev",
            "prompt": "a cat",
        },
    }
    noop = lambda *args, **kwargs: None  # noqa: E731
    try:
        FluxDiffusersAdapter().generate(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )
    except RuntimeError as exc:
        assert "does not support image editing" in str(exc)
    else:
        raise AssertionError("FLUX.1 is text-to-image only and must reject edit_image.")


def test_flux_adapter_applies_flux_lora(tmp_path):
    lora = tmp_path / "flux_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = FluxDiffusersAdapter()
    request = SimpleNamespace(
        model="flux_schnell",
        loras=[
            {
                "id": "flux_style",
                "installedPath": str(lora),
                "weight": 0.7,
                "compatibility": {"families": ["flux"]},
            }
        ],
    )
    adapter._apply_loras(pipe, request)
    state = adapter._loaded_lora_states["text"]
    assert [path for path, _name in pipe.loaded] == [str(lora)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.7])]


def test_flux_adapter_rejects_incompatible_lora_family(tmp_path):
    lora = tmp_path / "qwen_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = FluxDiffusersAdapter()
    request = SimpleNamespace(
        model="flux_dev",
        loras=[
            {
                "id": "qwen_style",
                "installedPath": str(lora),
                "compatibility": {"families": ["qwen-image"]},
            }
        ],
    )
    try:
        adapter._apply_loras(pipe, request)
    except RuntimeError as exc:
        assert "not compatible with model family flux" in str(exc)
    else:
        raise AssertionError("FLUX.1 must reject a LoRA whose family is not flux.")


def test_create_image_adapter_routes_chroma_variants():
    for model in ("chroma1_hd", "chroma1_base", "chroma1_flash"):
        adapter = create_image_adapter({"payload": {"model": model}})
        assert adapter.__class__.__name__ == "ChromaDiffusersAdapter"
        assert adapter.id == "chroma_diffusers"


def test_image_adapter_env_override_selects_chroma(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "chroma_diffusers")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "ChromaDiffusersAdapter"


def test_chroma_model_target_defaults():
    hd = MODEL_TARGETS["chroma1_hd"]
    base = MODEL_TARGETS["chroma1_base"]
    flash = MODEL_TARGETS["chroma1_flash"]
    for target in (hd, base, flash):
        assert target["adapter"] == "chroma_diffusers"
        assert target["family"] == "chroma"
        assert target["supportsEdit"] is False
        assert target["maxSequenceLength"] == 512
    # HD/Base: real CFG with negative prompts (~40 steps, guidance 3.0).
    assert hd["steps"] == 40
    assert hd["guidanceScale"] == 3.0
    assert hd["repo"] == "lodestones/Chroma1-HD"
    assert base["steps"] == 40
    assert base["guidanceScale"] == 3.0
    assert base["repo"] == "lodestones/Chroma1-Base"
    # Flash: CFG baked off (~8 steps, guidance 1.0).
    assert flash["steps"] == 8
    assert flash["guidanceScale"] == 1.0
    assert flash["repo"] == "lodestones/Chroma1-Flash"


def test_chroma_guidance_scale_uses_per_model_default_and_override():
    adapter = ChromaDiffusersAdapter()
    hd = MODEL_TARGETS["chroma1_hd"]
    flash = MODEL_TARGETS["chroma1_flash"]
    # Per-model default applies when the request does not override guidance.
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), hd) == 3.0
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), flash) == 1.0
    # An explicit request value wins.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 4.5}), hd) == 4.5
    # Unparseable override falls back to the per-model default.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": "x"}), hd) == 3.0


def test_chroma_num_inference_steps_default_and_override():
    adapter = ChromaDiffusersAdapter()
    hd = MODEL_TARGETS["chroma1_hd"]
    flash = MODEL_TARGETS["chroma1_flash"]
    # Defaults come straight from the model target.
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), hd) == 40
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), flash) == 8
    # Explicit override is honored and clamped to [1, 80].
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 20}), hd) == 20
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), flash) == 80


def test_chroma_rejects_image_edit():
    job = {
        "id": "job_chroma_edit",
        "payload": {
            "projectId": "project_x",
            "mode": "edit_image",
            "model": "chroma1_hd",
            "prompt": "a cat",
        },
    }
    noop = lambda *args, **kwargs: None  # noqa: E731
    try:
        ChromaDiffusersAdapter().generate(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )
    except RuntimeError as exc:
        assert "does not support image editing" in str(exc)
    else:
        raise AssertionError("Chroma1 is text-to-image only and must reject edit_image.")


def test_chroma_adapter_applies_chroma_lora(tmp_path):
    lora = tmp_path / "chroma_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = ChromaDiffusersAdapter()
    request = SimpleNamespace(
        model="chroma1_hd",
        loras=[
            {
                "id": "chroma_style",
                "installedPath": str(lora),
                "weight": 0.7,
                "compatibility": {"families": ["chroma"]},
            }
        ],
    )
    adapter._apply_loras(pipe, request)
    state = adapter._loaded_lora_states["text"]
    assert [path for path, _name in pipe.loaded] == [str(lora)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.7])]


def test_chroma_adapter_applies_flux_lora(tmp_path):
    # Chroma is FLUX.1-schnell-derived: Flux LoRAs (and Chroma LoRAs detected as
    # flux by their identical tensor keys) load on Chroma. sc-1832.
    lora = tmp_path / "flux_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = ChromaDiffusersAdapter()
    request = SimpleNamespace(
        model="chroma1_hd",
        loras=[
            {
                "id": "flux_style",
                "installedPath": str(lora),
                "weight": 0.8,
                "compatibility": {"families": ["flux"]},
            }
        ],
    )
    adapter._apply_loras(pipe, request)
    state = adapter._loaded_lora_states["text"]
    assert [path for path, _name in pipe.loaded] == [str(lora)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.8])]


def test_chroma_adapter_rejects_incompatible_lora_family(tmp_path):
    lora = tmp_path / "qwen_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = ChromaDiffusersAdapter()
    request = SimpleNamespace(
        model="chroma1_base",
        loras=[
            {
                "id": "qwen_style",
                "installedPath": str(lora),
                "compatibility": {"families": ["qwen-image"]},
            }
        ],
    )
    try:
        adapter._apply_loras(pipe, request)
    except RuntimeError as exc:
        assert "not compatible with model family chroma" in str(exc)
    else:
        raise AssertionError("Chroma1 must reject a LoRA whose family is neither chroma nor flux.")


def test_create_image_adapter_routes_kolors():
    adapter = create_image_adapter({"payload": {"model": "kolors"}})
    assert adapter.__class__.__name__ == "KolorsDiffusersAdapter"
    assert adapter.id == "kolors_diffusers"


def test_image_adapter_env_override_selects_kolors(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "kolors_diffusers")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "KolorsDiffusersAdapter"


def test_kolors_model_target_defaults():
    kolors = MODEL_TARGETS["kolors"]
    assert kolors["adapter"] == "kolors_diffusers"
    assert kolors["family"] == "kolors"
    # Unified checkpoint does both T2I (KolorsPipeline) and img2img edit.
    assert kolors["supportsEdit"] is True
    # Real CFG (not distilled): guidance 5.0, ~25 steps, ChatGLM3 max_seq_len 256.
    assert kolors["steps"] == 25
    assert kolors["guidanceScale"] == 5.0
    assert kolors["maxSequenceLength"] == 256
    # fp16-only repo: from_pretrained must request the fp16 variant.
    assert kolors["variant"] == "fp16"
    assert kolors["repo"] == "Kwai-Kolors/Kolors-diffusers"


def test_kolors_guidance_scale_uses_per_model_default_and_override():
    adapter = KolorsDiffusersAdapter()
    kolors = MODEL_TARGETS["kolors"]
    # Unlike the guidance-distilled FLUX path, Kolors defaults to real CFG (5.0).
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), kolors) == 5.0
    # An explicit request value wins.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 7.0}), kolors) == 7.0
    # Unparseable override falls back to the per-model default.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": "x"}), kolors) == 5.0


def test_kolors_num_inference_steps_default_and_override():
    adapter = KolorsDiffusersAdapter()
    kolors = MODEL_TARGETS["kolors"]
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), kolors) == 25
    # Explicit override is honored and clamped to [1, 80].
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 30}), kolors) == 30
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), kolors) == 80


def test_kolors_max_sequence_length_default_and_override():
    adapter = KolorsDiffusersAdapter()
    kolors = MODEL_TARGETS["kolors"]
    assert adapter._max_sequence_length(SimpleNamespace(advanced={}), kolors) == 256
    # Override honored, clamped to the ChatGLM3 max of 256.
    assert adapter._max_sequence_length(SimpleNamespace(advanced={"maxSequenceLength": 128}), kolors) == 128
    assert adapter._max_sequence_length(SimpleNamespace(advanced={"maxSequenceLength": 4096}), kolors) == 256


def test_kolors_supports_edit():
    # Kolors is a unified checkpoint: KolorsPipeline (T2I) + KolorsImg2ImgPipeline (edit).
    assert model_supports_edit("kolors") is True


def test_create_image_adapter_routes_kolors_edit():
    # Edit jobs route to the same adapter; the adapter switches pipeline by mode.
    adapter = create_image_adapter({"payload": {"model": "kolors", "mode": "edit_image"}})
    assert adapter.__class__.__name__ == "KolorsDiffusersAdapter"
    assert adapter.id == "kolors_diffusers"


def test_kolors_reference_asset_id_parsed():
    request = image_request_from_job(
        {"payload": {"projectId": "p", "model": "kolors", "referenceAssetId": "asset-ref"}}
    )
    assert request.reference_asset_id == "asset-ref"


def test_kolors_use_ip_adapter_only_for_text_with_reference():
    use = KolorsDiffusersAdapter._use_ip_adapter
    # IP-Adapter runs on the T2I pipeline with a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id="a")) is True
    # Not for edit jobs (reference + img2img together is a future enhancement)...
    assert use(SimpleNamespace(mode="edit_image", reference_asset_id="a")) is False
    # ...and not without a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id=None)) is False


def test_kolors_ip_adapter_scale_default_and_clamp():
    adapter = KolorsDiffusersAdapter()
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={})) == 0.6
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 0.4})) == 0.4
    # Clamped to [0, 1]; unparseable falls back to the default.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 5})) == 1.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": -2})) == 0.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": "x"})) == 0.6


def test_load_reference_image_requires_asset(tmp_path):
    try:
        load_reference_image(tmp_path, "")
    except RuntimeError as exc:
        assert "reference image asset" in str(exc)
    else:
        raise AssertionError("load_reference_image must reject a missing reference asset id.")


def test_kolors_reference_run_pipeline_passes_ip_adapter_image(tmp_path, monkeypatch):
    """A T2I job with referenceAssetId drives the IP-Adapter branch of _run_pipeline:
    load_reference_image(project_path, reference_asset_id) → ip_adapter_image kwarg, plus
    a per-request set_ip_adapter_scale. Mirrors test_edit_run_pipeline_threads_project_path
    so it runs torch-free in CI."""

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.scales: list[float] = []

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen: list[tuple] = []

    def fake_load_reference_image(project_path, reference_asset_id):
        seen.append((project_path, reference_asset_id))
        return FakeImage()

    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image", fake_load_reference_image
    )

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_image",
                "model": "kolors",
                "prompt": "a portrait of the character",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {"ipAdapterScale": 0.7},
            }
        }
    )
    pipe = FakePipe()
    result = KolorsDiffusersAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"), pipe, request, 7, project_path
    )
    # The IP-Adapter branch ran: it loaded the reference image (→ ip_adapter_image
    # kwarg) and applied the per-request scale. (filter_call_kwargs only keeps a
    # pipe's *named* params, and FakePipe takes **kwargs, so we assert via the
    # observable side effects rather than last_kwargs — same as the edit test.)
    assert seen == [(project_path, "asset-ref")]
    assert pipe.scales == [0.7]
    assert result is FakeOutput.images[0]


def test_kolors_pose_entries_gating():
    """sc-2264: the pose ControlNet path engages only for a character_image job WITH a
    reference and advanced.poses — not for edit_image, and not without a reference."""
    entries = KolorsDiffusersAdapter._pose_entries
    kp = [{"id": "sit", "keypoints": [[0.5, 0.5]] * 18}]
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={"poses": kp})) == kp
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id=None, advanced={"poses": kp})) == []
    assert entries(SimpleNamespace(mode="edit_image", reference_asset_id="r", advanced={"poses": kp})) == []
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={})) == []


def test_kolors_run_pose_composes_skeleton_controlnet_and_ip_adapter(tmp_path, monkeypatch):
    """sc-2264 strict tier: _run_pose feeds the rendered skeleton as control_image, the
    reference as both ip_adapter_image (identity) and img2img init, and the openPoseScale
    as controlnet_conditioning_scale — pose ControlNet + IP-Adapter in one call."""
    import numpy as _np
    from scene_worker import image_adapters as ia

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.last_kwargs: dict = {}
            self.scales: list = []

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        def __call__(self, *, prompt=None, image=None, control_image=None, ip_adapter_image=None,
                     controlnet_conditioning_scale=None, **kwargs):
            self.last_kwargs = {
                "image": image, "control_image": control_image, "ip_adapter_image": ip_adapter_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale, **kwargs,
            }
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    skeleton_sentinel = FakeImage()
    reference_sentinel = FakeImage()
    # Kolors _run_pose renders the DWPose whole-body skeleton (sc-2289); draw_wholebody
    # needs cv2 (absent in the CI venv), so stub it (real render covered elsewhere).
    monkeypatch.setattr(
        ia,
        "draw_wholebody",
        lambda w, h, kps, hands=None, face=None, stickwidth=4: _np.zeros((h, w, 3), dtype=_np.uint8),
    )
    monkeypatch.setattr("scene_worker.image_adapters.Image", SimpleNamespace(fromarray=lambda arr: skeleton_sentinel))
    monkeypatch.setattr(ia, "load_reference_image", lambda project_path, asset_id: reference_sentinel)

    request = image_request_from_job({"payload": {
        "projectId": "p", "mode": "character_image", "model": "kolors",
        "prompt": "the character", "referenceAssetId": "asset-ref",
        "width": 16, "height": 16, "count": 1, "advanced": {"openPoseScale": 0.65, "ipAdapterScale": 0.6},
    }})
    pipe = FakePipe()
    kp = [(0.5, 0.1 + 0.04 * i) for i in range(18)]
    KolorsDiffusersAdapter()._run_pose(SimpleNamespace(gpu_id="cpu"), pipe, request, 7, tmp_path, kp)
    assert pipe.last_kwargs["control_image"] is skeleton_sentinel  # pose -> ControlNet
    assert pipe.last_kwargs["ip_adapter_image"] is reference_sentinel  # identity -> IP-Adapter
    assert pipe.last_kwargs["image"] is reference_sentinel  # img2img init = reference
    assert pipe.last_kwargs["controlnet_conditioning_scale"] == 0.65
    assert pipe.scales == [0.6]


def test_kolors_pose_set_loops_poses_with_shared_seed(tmp_path, monkeypatch):
    """sc-2264: generate() with advanced.poses routes to the ControlNet pose path — one
    image per library pose, a single shared seed, and poseLibrary recorded in settings."""
    from scene_worker import image_adapters as ia

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_load_pose_pipeline", lambda self, *a, **k: object())
    monkeypatch.setattr(ia, "select_torch_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: SimpleNamespace() if name == "torch" else importlib.import_module(name),
    )

    calls: list[dict] = []

    def fake_run_pose(self, settings, pipe, request, seed, project_path, keypoints, hands=None, face=None, cancel_requested=None):
        from PIL import Image as _Image
        calls.append({"seed": seed, "keypoints": keypoints, "hands": hands, "face": face})
        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_run_pose", fake_run_pose)

    captured: dict = {}

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, raw_settings, **kwargs):
            captured["raw_settings"] = raw_settings
            captured["image_count"] = image_count
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)

    kp = [[0.5, 0.1 + 0.04 * i] for i in range(18)]
    hands = [[[0.4, 0.4]] * 21, [[0.6, 0.4]] * 21]
    face = [[0.5, 0.3]] * 68
    job = {"id": "job_kolors_pose", "payload": {
        "projectId": "p", "mode": "character_image", "model": "kolors", "prompt": "the character",
        "referenceAssetId": "ref-1", "count": 1, "width": 64, "height": 64,
        "advanced": {"poses": [
            {"id": "sit_01", "keypoints": kp},  # body-only
            {"id": "dance_01", "keypoints": kp, "hands": hands, "face": face},  # whole-body
        ]},
    }}
    KolorsDiffusersAdapter().generate(
        settings=SimpleNamespace(gpu_id="cpu"), job=job, request=image_request_from_job(job),
        project_path=tmp_path, progress=lambda *a, **k: None, cancel_requested=lambda: False,
    )
    assert captured["image_count"] == 2  # one per pose, not request.count
    assert len(calls) == 2
    assert calls[0]["seed"] == calls[1]["seed"]  # shared seed across the set
    # sc-2289: whole-body hands/face thread through to the DWPose-trained Kolors CN;
    # body-only poses pass None (rendered identically to the old body-only path).
    assert calls[0]["hands"] is None and calls[0]["face"] is None
    assert calls[1]["hands"] is not None and len(calls[1]["face"]) == 68
    assert captured["raw_settings"].get("poseLibrary") is True
    assert captured["raw_settings"].get("controlNetPose") == "Kwai-Kolors/Kolors-ControlNet-Pose"


def test_kolors_pose_set_applies_loras_once_before_loop(tmp_path, monkeypatch):
    """sc-2251/sc-2252: the strict Kolors pose tier must apply request.loras exactly
    once on the pose pipe BEFORE the per-pose loop (the same merge the T2I/img2img
    path does), under its own "pose" cache slot — so the character-LoRA bootstrapping
    loop works on pose generation. Guards the regression where _generate_pose_set
    returned before the LoRA merge, silently ignoring request.loras (and LoKr)."""
    from scene_worker import image_adapters as ia

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_load_pose_pipeline", lambda self, *a, **k: object())
    monkeypatch.setattr(ia, "select_torch_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: SimpleNamespace() if name == "torch" else importlib.import_module(name),
    )

    apply_calls: list[dict] = []

    def fake_apply_loras(self, pipe, request, *, lora_key=None):
        apply_calls.append({"loras": list(request.loras), "lora_key": lora_key})

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_apply_loras", fake_apply_loras)

    def fake_run_pose(self, settings, pipe, request, seed, project_path, keypoints, hands=None, face=None, cancel_requested=None):
        from PIL import Image as _Image

        assert apply_calls, "loras must be applied before the pose loop runs"
        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_run_pose", fake_run_pose)

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, **kwargs):
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)

    kp = [[0.5, 0.1 + 0.04 * i] for i in range(18)]
    loras = [{"id": "kelsie", "path": "/loras/kelsie.safetensors", "families": ["kolors"]}]
    job = {"id": "job_kolors_pose_lora", "payload": {
        "projectId": "p", "mode": "character_image", "model": "kolors", "prompt": "the character",
        "referenceAssetId": "ref-1", "count": 1, "width": 64, "height": 64, "loras": loras,
        "advanced": {"poses": [{"id": "sit_01", "keypoints": kp}, {"id": "stand_01", "keypoints": kp}]},
    }}
    KolorsDiffusersAdapter().generate(
        settings=SimpleNamespace(gpu_id="cpu"), job=job, request=image_request_from_job(job),
        project_path=tmp_path, progress=lambda *a, **k: None, cancel_requested=lambda: False,
    )
    # Applied exactly once (not once-per-pose), before the loop, under the dedicated
    # "pose" cache slot so it never collides with the text/img2img pipe bookkeeping.
    assert len(apply_calls) == 1
    assert apply_calls[0]["loras"] == loras
    assert apply_calls[0]["lora_key"] == "pose"


def test_kolors_pose_path_accepts_lokr_via_injection(monkeypatch, tmp_path):
    """sc-2252: the Kolors strict pose tier must APPLY LoKr (not just plain LoRA) on its
    torch pose pipe. _apply_loras(lora_key="pose") routes lokr_* through inject_lokr_adapter
    into the pose pipe's UNet — never load_lora_weights — and tracks the merge under the
    dedicated "pose" cache slot, distinct from the text/img2img bookkeeping."""
    lokr_file = tmp_path / "char.safetensors"
    lokr_file.write_bytes(b"x")
    monkeypatch.setattr("scene_worker.lora_adapters.adapter_network_type", lambda path: "lokr")
    injected: list[str] = []
    monkeypatch.setattr(
        "scene_worker.lora_adapters.inject_lokr_adapter",
        lambda pipe, spec, *, adapter_id: injected.append(spec.adapter_name),
    )
    pipe = FakeLokrPipe()
    adapter = KolorsDiffusersAdapter()
    request = SimpleNamespace(
        mode="character_image",
        model="kolors",
        loras=[{"id": "char", "installedPath": str(lokr_file), "weight": 0.7, "families": ["kolors"]}],
    )

    adapter._apply_loras(pipe, request, lora_key="pose")

    # LoKr injects into the pose pipe's UNet denoiser; it never loads via load_lora_weights.
    assert pipe.loaded == []
    assert injected, "LoKr must inject into the pose pipe denoiser"
    # Tracked under the dedicated pose slot — not the text/img2img slots.
    pose_state = adapter._loaded_lora_states["pose"]
    assert injected == list(pose_state.adapter_names)
    assert "text" not in adapter._loaded_lora_states
    assert "img2img" not in adapter._loaded_lora_states


def test_create_image_adapter_routes_sdxl(monkeypatch):
    # SDXL resolves to the torch adapter on the Python worker (sc-3060 retired the
    # vendored MLX SDXL path; Mac MLX routing now lives in the Rust API claim layer).
    monkeypatch.delenv("SCENEWORKS_IMAGE_ADAPTER", raising=False)
    adapter = create_image_adapter({"payload": {"model": "sdxl"}})
    assert adapter.__class__.__name__ == "SdxlDiffusersAdapter"
    assert adapter.id == "sdxl_diffusers"


def test_image_adapter_env_override_selects_sdxl(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "sdxl_diffusers")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "SdxlDiffusersAdapter"


def test_sdxl_model_target_defaults():
    sdxl = MODEL_TARGETS["sdxl"]
    assert sdxl["adapter"] == "sdxl_diffusers"
    assert sdxl["family"] == "sdxl"
    # Unified base checkpoint does both T2I and img2img edit.
    assert sdxl["supportsEdit"] is True
    # Real CFG with negative prompt: ~30 steps at guidance 7.0.
    assert sdxl["steps"] == 30
    assert sdxl["guidanceScale"] == 7.0
    # fp16 variant; two CLIP encoders so there is no max_sequence_length knob.
    assert sdxl["variant"] == "fp16"
    assert "maxSequenceLength" not in sdxl
    assert sdxl["repo"] == "stabilityai/stable-diffusion-xl-base-1.0"
    # IP-Adapter plus-face for Character Studio reference conditioning (sc-2007):
    # ViT-H image encoder shipped in the same repo, sdxl_models subfolder for the
    # weights. Plus-face is the identity-leaning variant without scene/outfit copy.
    ip_adapter = sdxl["ipAdapter"]
    assert ip_adapter["repo"] == "h94/IP-Adapter"
    assert ip_adapter["subfolder"] == "sdxl_models"
    assert ip_adapter["weight"] == "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    assert ip_adapter["encoderSubfolder"] == "models/image_encoder"


def test_sdxl_supports_edit():
    # SDXL base is a unified checkpoint: StableDiffusionXLPipeline (T2I) +
    # StableDiffusionXLImg2ImgPipeline (edit).
    assert model_supports_edit("sdxl") is True


def test_create_image_adapter_routes_sdxl_edit():
    # Edit jobs route to the same adapter; it switches pipeline by mode.
    adapter = create_image_adapter({"payload": {"model": "sdxl", "mode": "edit_image"}})
    assert adapter.__class__.__name__ == "SdxlDiffusersAdapter"
    assert adapter.id == "sdxl_diffusers"


def test_sdxl_guidance_scale_uses_per_model_default_and_override():
    adapter = SdxlDiffusersAdapter()
    sdxl = MODEL_TARGETS["sdxl"]
    # SDXL uses real CFG; the per-model default (7.0) applies without an override.
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), sdxl) == 7.0
    # An explicit request value wins.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 5.0}), sdxl) == 5.0
    # Unparseable override falls back to the per-model default.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": "x"}), sdxl) == 7.0


def test_zimage_num_inference_steps_matches_manifest_default():
    # sc-4188 drift fix: ZImage defaulted to model_target["steps"] + 1 (9) with
    # no rationale while every other adapter — and the MLX engine path — uses
    # the manifest value unmodified (8).
    from scene_worker.image_adapters import ZImageDiffusersAdapter

    adapter = ZImageDiffusersAdapter()
    target = MODEL_TARGETS["z_image_turbo"]
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), target) == target["steps"] == 8
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 12}), target) == 12


def test_generate_images_always_wires_upscaler_telemetry(monkeypatch):
    # sc-4188 drift fix: five adapters dropped settings=/job_id= from the
    # incremental writer call, silently disabling upscaler telemetry/job-id
    # wiring. The shared _generate_images passes them unconditionally.
    from scene_worker.image_adapters import FluxDiffusersAdapter, ImageAssetWriter

    captured: dict = {}

    def fake_writer(self, **kwargs):
        captured.update(kwargs)
        return {"images": []}

    monkeypatch.setattr(ImageAssetWriter, "write_incremental_outputs", fake_writer)
    fake_torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
                                 cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: fake_torch if name == "torch" else importlib.import_module(name),
    )
    adapter = FluxDiffusersAdapter()
    settings = SimpleNamespace(gpu_id="cpu")
    request = SimpleNamespace(model="flux_schnell", seed=1, seeds=[], prompt="p", advanced={})
    adapter._generate_images(
        settings=settings,
        job={"id": "job-telemetry"},
        request=request,
        project_path=Path("."),
        progress=lambda *a, **k: None,
        cancel_requested=lambda: False,
        label="FLUX.1",
        image_count=1,
        run_one=lambda index, seed: FakeImage(),
        raw_settings={"realModelInference": True},
    )
    assert captured["settings"] is settings
    assert captured["job_id"] == "job-telemetry"


def test_sdxl_num_inference_steps_default_and_override():
    adapter = SdxlDiffusersAdapter()
    sdxl = MODEL_TARGETS["sdxl"]
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), sdxl) == 30
    # Explicit override is honored and clamped to [1, 80].
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 45}), sdxl) == 45
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), sdxl) == 80


def test_sdxl_reference_asset_id_parsed():
    request = image_request_from_job(
        {"payload": {"projectId": "p", "model": "sdxl", "referenceAssetId": "asset-ref"}}
    )
    assert request.reference_asset_id == "asset-ref"


def test_sdxl_use_ip_adapter_only_for_text_with_reference():
    use = SdxlDiffusersAdapter._use_ip_adapter
    # IP-Adapter runs on the T2I pipeline with a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id="a")) is True
    # Not for edit jobs (reference + img2img together is a future enhancement)...
    assert use(SimpleNamespace(mode="edit_image", reference_asset_id="a")) is False
    # ...and not without a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id=None)) is False


def test_sdxl_ip_adapter_scale_default_and_clamp():
    adapter = SdxlDiffusersAdapter()
    # Plus-face default is 0.7 — identity-leaning while letting the prompt steer.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={})) == 0.7
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 0.4})) == 0.4
    # Clamped to [0, 1]; unparseable falls back to the default.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 5})) == 1.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": -2})) == 0.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": "x"})) == 0.7


def test_sdxl_reference_run_pipeline_passes_ip_adapter_image(tmp_path, monkeypatch):
    """A T2I job with referenceAssetId drives the IP-Adapter branch of _run_pipeline:
    load_reference_image(project_path, reference_asset_id) → ip_adapter_image kwarg, plus
    a per-request set_ip_adapter_scale. Mirrors test_kolors_reference_run_pipeline_passes_ip_adapter_image
    so it runs torch-free in CI."""

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.scales: list[float] = []

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen: list[tuple] = []

    def fake_load_reference_image(project_path, reference_asset_id):
        seen.append((project_path, reference_asset_id))
        return FakeImage()

    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image", fake_load_reference_image
    )

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_image",
                "model": "sdxl",
                "prompt": "a portrait of the character",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {"ipAdapterScale": 0.5},
            }
        }
    )
    pipe = FakePipe()
    result = SdxlDiffusersAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"), pipe, request, 7, project_path
    )
    # The IP-Adapter branch ran: it loaded the reference image (→ ip_adapter_image
    # kwarg) and applied the per-request scale. (filter_call_kwargs only keeps a
    # pipe's *named* params, and FakePipe takes **kwargs, so we assert via the
    # observable side effects rather than last_kwargs — same as the edit test.)
    assert seen == [(project_path, "asset-ref")]
    assert pipe.scales == [0.5]
    assert result is FakeOutput.images[0]


def test_create_image_adapter_routes_realvisxl():
    # RealVisXL is a photoreal SDXL finetune that rides the same adapter; pickers
    # filter by capability flag, not adapter id (sc-2008).
    adapter = create_image_adapter({"payload": {"model": "realvisxl"}})
    assert adapter.__class__.__name__ == "SdxlDiffusersAdapter"
    assert adapter.id == "sdxl_diffusers"


def test_realvisxl_model_target_defaults():
    target = MODEL_TARGETS["realvisxl"]
    # Same adapter + family as sdxl (sdxl-family LoRAs apply); plain photoreal
    # selectable that complements instantid_realvisxl on the same checkpoint.
    assert target["adapter"] == "sdxl_diffusers"
    assert target["family"] == "sdxl"
    assert target["supportsEdit"] is True
    # SDXL defaults: ~30 steps at guidance 7.0, fp16 variant.
    assert target["steps"] == 30
    assert target["guidanceScale"] == 7.0
    assert target["variant"] == "fp16"
    # RealVisXL_V5.0: photoreal SDXL finetune (openrail++, ungated). Shares the
    # HF cache with the InstantID built-in (no duplicate download).
    assert target["repo"] == "SG161222/RealVisXL_V5.0"
    # IP-Adapter block matches sdxl: any SDXL-UNet checkpoint can reuse it.
    assert target["ipAdapter"] == MODEL_TARGETS["sdxl"]["ipAdapter"]


def test_realvisxl_supports_edit():
    assert model_supports_edit("realvisxl") is True


def test_sdxl_adapter_applies_sdxl_lora(tmp_path):
    # A trained sdxl-family LoRA loads onto the SDXL pipeline via
    # StableDiffusionXLPipeline.load_lora_weights and its weight is applied (sc-1943).
    lora = tmp_path / "aurora_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = SdxlDiffusersAdapter()
    request = SimpleNamespace(
        model="sdxl",
        mode="text_to_image",
        loras=[
            {
                "id": "aurora_style",
                "installedPath": str(lora),
                "weight": 0.75,
                "compatibility": {"families": ["sdxl"]},
            }
        ],
    )
    adapter._apply_loras(pipe, request)
    state = adapter._loaded_lora_states["text"]
    assert [path for path, _name in pipe.loaded] == [str(lora)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.75])]


def test_sdxl_adapter_rejects_incompatible_lora_family(tmp_path):
    # SDXL accepts only sdxl-family LoRAs (no extra-compatible families, unlike
    # chroma↔flux): a flux LoRA is filtered out before it can load (sc-1927/sc-1943).
    lora = tmp_path / "flux_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = SdxlDiffusersAdapter()
    request = SimpleNamespace(
        model="sdxl",
        mode="text_to_image",
        loras=[
            {
                "id": "flux_style",
                "installedPath": str(lora),
                "compatibility": {"families": ["flux"]},
            }
        ],
    )
    try:
        adapter._apply_loras(pipe, request)
    except RuntimeError as exc:
        assert "not compatible with model family sdxl" in str(exc)
    else:
        raise AssertionError("SDXL must reject a LoRA whose family is not sdxl.")


def test_sdxl_backend_save_lora_writes_unet_diffusers_format(tmp_path, monkeypatch):
    # The trainer saves an sdxl-family LoRA via StableDiffusionXLPipeline's own
    # save_lora_weights(unet_lora_layers=...) — the diffusers format the inference
    # loader (load_lora_weights) round-trips. Torch-free: fake peft + pipeline.
    import sys
    import types as types_module

    fake_state = {"unet.down_blocks.0.attentions.0.lora_A.weight": "tensor"}
    fake_peft_utils = types_module.ModuleType("peft.utils")
    fake_peft_utils.get_peft_model_state_dict = lambda _module: fake_state
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)

    saved: dict[str, object] = {}

    class FakeSdxlPipeline:
        @staticmethod
        def save_lora_weights(
            output_dir, *, unet_lora_layers=None, weight_name=None, safe_serialization=None, **_kwargs
        ):
            saved["output_dir"] = output_dir
            saved["unet_lora_layers"] = unet_lora_layers
            saved["weight_name"] = weight_name
            saved["safe_serialization"] = safe_serialization

    backend = _SdxlLoraBackend()
    backend._unet = object()
    backend._pipeline = FakeSdxlPipeline()
    output_path = backend._save_lora(output_dir=str(tmp_path), file_name="aurora_style.safetensors")

    # unet_lora_layers carries the PEFT state dict; this is the only LoRA layer
    # kwarg the trainer passes (UNet-only), and the key the SDXL loader consumes.
    assert saved["unet_lora_layers"] is fake_state
    assert saved["weight_name"] == "aurora_style.safetensors"
    assert saved["safe_serialization"] is True
    assert output_path == str(tmp_path / "aurora_style.safetensors")


def test_wan_backend_save_lora_writes_transformer_diffusers_format(tmp_path, monkeypatch):
    # Default (lora) network: the Wan trainer saves via WanPipeline.save_lora_weights
    # (transformer_lora_layers=...), the diffusers format the video loader round-trips.
    # Torch-free: fake peft + pipeline (sc-2211).
    import sys
    import types as types_module

    fake_state = {"transformer.blocks.0.attn1.to_q.lora_A.weight": "tensor"}
    fake_peft_utils = types_module.ModuleType("peft.utils")
    fake_peft_utils.get_peft_model_state_dict = lambda _module: fake_state
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)

    saved: dict[str, object] = {}

    class FakeWanPipeline:
        @staticmethod
        def save_lora_weights(
            output_dir, *, transformer_lora_layers=None, weight_name=None, safe_serialization=None, **_kwargs
        ):
            saved["transformer_lora_layers"] = transformer_lora_layers
            saved["weight_name"] = weight_name
            saved["safe_serialization"] = safe_serialization

    backend = _WanLoraBackend()
    backend._transformer = object()
    backend._pipeline = FakeWanPipeline()
    output_path = backend._save_lora(output_dir=str(tmp_path), file_name="motion.safetensors")

    assert saved["transformer_lora_layers"] is fake_state
    assert saved["weight_name"] == "motion.safetensors"
    assert saved["safe_serialization"] is True
    assert output_path == str(tmp_path / "motion.safetensors")


def test_wan_backend_save_lora_lokr_routes_to_write_lokr_adapter(tmp_path, monkeypatch):
    # LoKr network (sc-2211): LoKr keys (lokr_w1/lokr_w2) aren't save_lora_weights-
    # compatible, so the Wan trainer serializes raw via write_lokr_adapter with the
    # routing metadata the video inference loader (PEFT injection) needs — exactly
    # like the SDXL/Z-Image backends. save_lora_weights must NOT be called.
    import sys
    import types as types_module

    fake_state = {"transformer.blocks.0.attn1.to_q.lokr_w1": "tensor"}
    fake_peft_utils = types_module.ModuleType("peft.utils")
    fake_peft_utils.get_peft_model_state_dict = lambda _module: fake_state
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)

    captured: dict[str, object] = {}

    def fake_write_lokr_adapter(state_dict, output_dir, file_name, **kwargs):
        captured["state_dict"] = state_dict
        captured["file_name"] = file_name
        captured["kwargs"] = kwargs
        return str(Path(output_dir) / file_name)

    monkeypatch.setattr(
        "scene_worker.training_adapters.write_lokr_adapter", fake_write_lokr_adapter
    )

    class FakeWanPipeline:
        @staticmethod
        def save_lora_weights(*_args, **_kwargs):
            raise AssertionError("LoKr must not save via save_lora_weights")

    backend = _WanLoraBackend()
    backend._transformer = object()
    backend._pipeline = FakeWanPipeline()
    backend._network_type = "lokr"
    save_kwargs = {
        "rank": 16,
        "alpha": 16,
        "decompose_factor": -1,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    }
    backend._lokr_save_kwargs = save_kwargs

    output_path = backend._save_lora(output_dir=str(tmp_path), file_name="motion.safetensors")

    assert captured["state_dict"] is fake_state
    assert captured["file_name"] == "motion.safetensors"
    assert captured["kwargs"] == save_kwargs
    assert output_path == str(tmp_path / "motion.safetensors")


def test_create_image_adapter_routes_sensenova_u1():
    adapter = create_image_adapter({"payload": {"model": "sensenova_u1_8b"}})
    assert adapter.__class__.__name__ == "SenseNovaU1Adapter"
    assert adapter.id == "sensenova_u1"


def test_image_adapter_env_override_selects_sensenova(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "sensenova_u1")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "SenseNovaU1Adapter"


def test_sensenova_u1_model_target_defaults():
    target = MODEL_TARGETS["sensenova_u1_8b"]
    assert target["adapter"] == "sensenova_u1"
    assert target["family"] == "sensenova-u1"
    assert target["steps"] == 50
    # Unified model: the base entry supports instruction editing (it2i).
    assert target["supportsEdit"] is True
    assert target["repo"] == "sensenova/SenseNova-U1-8B-MoT"


def test_sensenova_u1_edit_support():
    # Both the base unified model and the distilled fast variant support editing.
    assert model_supports_edit("sensenova_u1_8b") is True
    assert model_supports_edit("sensenova_u1_8b_fast") is True


# ---- sc-2016: SenseNova-U1 character_image (wardrobe-preserving reference) ----


def test_sensenova_u1_use_reference_gates_on_mode_and_asset():
    """The Character Studio reference path activates only when mode is
    `character_image` AND a referenceAssetId is present. Mutually exclusive with
    `edit_image` (mirrors QwenImageAdapter._use_reference)."""
    from types import SimpleNamespace
    use_ref = SenseNovaU1Adapter._use_reference
    assert use_ref(SimpleNamespace(mode="character_image", reference_asset_id="asset_x")) is True
    # No reference asset → falls back to plain t2i, never to the edit path with a None source.
    assert use_ref(SimpleNamespace(mode="character_image", reference_asset_id=None)) is False
    # Edit mode is its own branch; character_image gating must not catch it.
    assert use_ref(SimpleNamespace(mode="edit_image", reference_asset_id="asset_x")) is False
    # Plain text-to-image with a stray referenceAssetId stays t2i.
    assert use_ref(SimpleNamespace(mode="text_to_image", reference_asset_id="asset_x")) is False


def test_sensenova_u1_image_guidance_scale_default_pulls_for_character_image():
    """The image-conditioning guidance defaults to 1.0 for edit_image (upstream
    default) and 1.5 for character_image (pull harder toward the reference).
    Per-request overrides via advanced.imageGuidanceScale win in both modes."""
    from types import SimpleNamespace
    img_cfg = SenseNovaU1Adapter._image_guidance_scale
    # Default per-mode:
    assert img_cfg(SimpleNamespace(advanced={})) == 1.0  # edit baseline
    assert img_cfg(SimpleNamespace(advanced={}), default=1.5) == 1.5  # character_image default
    # Override wins regardless of the default:
    assert img_cfg(SimpleNamespace(advanced={"imageGuidanceScale": 2.5}), default=1.5) == 2.5
    # Unparseable values fall back to the supplied default.
    assert img_cfg(SimpleNamespace(advanced={"imageGuidanceScale": "x"}), default=1.5) == 1.5


def test_sensenova_u1_angle_set_loops_augmented_prompts(tmp_path, monkeypatch):
    """SenseNova angle sets must generate one image per canonical angle using
    the per-angle prompt augment. The practical Character Studio picker exposes
    the fast target, but both targets share this adapter path."""
    from scene_worker import image_adapters as ia
    from scene_worker.character_studio_angles import ANGLE_PROMPT_AUGMENTS

    class FakeTorch:
        pass

    adapter = SenseNovaU1Adapter()
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr(ia, "require_inference_backend_for_gpu_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "select_torch_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ia, "activate_torch_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "select_torch_dtype", lambda *args, **kwargs: "float32")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "load_reference_image", lambda *args, **kwargs: Image.new("RGB", (8, 8)))
    monkeypatch.setattr(adapter, "_load_model", lambda *args, **kwargs: (object(), object()))

    captured: list[dict[str, Any]] = []

    def fake_run_edit(
        torch,
        model,
        tokenizer,
        prompt,
        source_image,
        width,
        height,
        steps,
        guidance_scale,
        img_guidance_scale,
        timestep_shift,
        seed,
    ):
        captured.append(
            {
                "prompt": prompt,
                "sourceSize": source_image.size,
                "steps": steps,
                "guidanceScale": guidance_scale,
                "seed": seed,
            }
        )
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(adapter, "_run_edit_inference", fake_run_edit)

    writer_capture: dict[str, Any] = {}

    def fake_writer(self, *, image_count, image_at_index, raw_settings, **kwargs):
        writer_capture["image_count"] = image_count
        writer_capture["raw_settings"] = raw_settings
        for index in range(image_count):
            image_at_index(index)
        return {"images": [], "count": image_count}

    monkeypatch.setattr(ImageAssetWriter, "write_incremental_outputs", fake_writer)

    job = {
        "id": "job-sensenova-angle",
        "payload": {
            "projectId": "p",
            "mode": "character_image",
            "model": "sensenova_u1_8b_fast",
            "prompt": "the character",
            "referenceAssetId": "ref-1",
            "count": 1,
            "seed": 42,
            "width": 1024,
            "height": 1024,
            "advanced": {"angleSet": True},
        },
    }
    adapter.generate(
        settings=SimpleNamespace(gpu_id="cpu"),
        job=job,
        request=image_request_from_job(job),
        project_path=tmp_path,
        progress=lambda *a, **k: None,
        cancel_requested=lambda: False,
    )

    assert writer_capture["image_count"] == len(CHARACTER_ANGLE_SET_ORDER)
    assert writer_capture["raw_settings"]["angleSet"] is True
    assert writer_capture["raw_settings"]["numInferenceSteps"] == 8
    assert len(captured) == len(CHARACTER_ANGLE_SET_ORDER)
    assert {entry["seed"] for entry in captured} == {42}
    for entry, angle in zip(captured, CHARACTER_ANGLE_SET_ORDER):
        assert ANGLE_PROMPT_AUGMENTS[angle] in entry["prompt"]
        assert entry["sourceSize"] == (2048, 2048)
        assert entry["steps"] == 8
        assert entry["guidanceScale"] == 1.0


def test_sensenova_u1_advertises_character_image_capability():
    """Both sensenova_u1 targets must advertise `character_image` in the builtin
    manifest now that the worker dispatches it through the it2i_generate path
    (sc-2016). Without this the sc-2018 reverse-drift guard would mark the
    engine as 'wired but unreachable from the picker'."""
    manifest = _load_builtin_models_manifest()
    by_id = {model["id"]: model for model in manifest.get("models", [])}
    for model_id in ("sensenova_u1_8b", "sensenova_u1_8b_fast"):
        capabilities = by_id[model_id].get("capabilities") or []
        assert "character_image" in capabilities, (
            f"{model_id} must advertise character_image in the builtin manifest "
            f"so the Image Studio picker surfaces the SenseNova reference flow."
        )
        ui = by_id[model_id].get("ui") or {}
        # sc-2018 audit accepts ui.variationStrength as the edit-backbone engine
        # declaration. SenseNova has no IP-Adapter / face-ID engine block, so
        # this is the required gate for the character_image capability to be honest.
        assert ui.get("variationStrength"), (
            f"{model_id} declares character_image but no ui.variationStrength; "
            f"the sc-2018 audit will fail (engine declaration missing)."
        )
    base_angles = by_id["sensenova_u1_8b"].get("ui", {}).get("viewAngles") or []
    assert base_angles == [], "The 50-step base model must not be advertised for Character Studio angle sets."
    fast_angle_ids = {
        angle["id"] for angle in by_id["sensenova_u1_8b_fast"].get("ui", {}).get("viewAngles") or []
    }
    assert fast_angle_ids == set(CHARACTER_ANGLE_SET_ORDER), (
        "sensenova_u1_8b_fast must expose the canonical angle set in Character Studio; "
        f"missing={set(CHARACTER_ANGLE_SET_ORDER) - fast_angle_ids}, "
        f"extra={fast_angle_ids - set(CHARACTER_ANGLE_SET_ORDER)}"
    )


def test_sensenova_u1_vqa_strips_reasoning():
    strip = SenseNovaU1Adapter._strip_reasoning
    # A complete think block is removed, leaving only the answer.
    assert strip("<think>\nweigh the options\n</think>\n\nIt's nighttime in Paris.") == "It's nighttime in Paris."
    # A dangling/unclosed think block (reasoning truncated by the token cap) yields no leak.
    assert strip("<think>\nreasoning that got cut off mid-thought") == ""
    # A plain answer (no reasoning) is returned unchanged.
    assert strip("A woman holds a yellow umbrella.") == "A woman holds a yellow umbrella."


def test_sensenova_u1_vqa_requires_question():
    # The VQA entry point validates the question before any model load (no torch).
    job = {
        "id": "job_vqa",
        "payload": {"projectId": "p", "sourceAssetId": "asset_1", "question": "   ", "model": "sensenova_u1_8b"},
    }
    noop = lambda *args, **kwargs: None  # noqa: E731
    try:
        SenseNovaU1Adapter().answer_question(settings=None, job=job, progress=noop, cancel_requested=lambda: False)
    except RuntimeError as exc:
        assert "requires a question" in str(exc)
    else:
        raise AssertionError("VQA must reject an empty question.")


def test_interleave_resolution_for_snaps_to_interleave_buckets():
    # Interleave uses its own (smaller) trained buckets, distinct from t2i.
    assert interleave_resolution_for(1000, 1000) == (1536, 1536)
    assert interleave_resolution_for(1920, 1080) == (2048, 1152)
    assert interleave_resolution_for(1080, 1920) == (1152, 2048)
    # Same 16:9 request snaps differently for interleave vs t2i.
    assert interleave_resolution_for(1920, 1080) != sensenova_resolution_for(1920, 1080)


def test_generate_interleaved_requires_prompt():
    # The prompt is validated before any model load (no torch needed).
    adapter = SenseNovaU1Adapter()
    job = {"id": "job_il", "payload": {"projectId": "p", "prompt": "   "}}
    noop = lambda *args, **kwargs: None  # noqa: E731
    with pytest.raises(RuntimeError, match="requires a prompt"):
        adapter.generate_interleaved(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )


def test_generate_interleaved_rejects_non_sensenova_model():
    adapter = SenseNovaU1Adapter()
    job = {"id": "job_il", "payload": {"projectId": "p", "prompt": "guide", "model": "z_image_turbo"}}
    noop = lambda *args, **kwargs: None  # noqa: E731
    with pytest.raises(RuntimeError, match="not a SenseNova-U1 target"):
        adapter.generate_interleaved(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )


def test_interleave_segments_interleave_text_and_images():
    assets = [
        {"assetId": "asset_a", "mediaPath": "assets/images/a.png"},
        {"assetId": "asset_b", "mediaPath": "assets/images/b.png"},
    ]
    text = "Boil the water.<image>Steep the leaves.<image>Pour and enjoy."
    segments = SenseNovaU1Adapter._build_interleaved_segments(text, assets)
    assert segments == [
        {"type": "text", "text": "Boil the water."},
        {"type": "image", "assetId": "asset_a", "path": "assets/images/a.png"},
        {"type": "text", "text": "Steep the leaves."},
        {"type": "image", "assetId": "asset_b", "path": "assets/images/b.png"},
        {"type": "text", "text": "Pour and enjoy."},
    ]


def test_interleave_segments_handles_leading_image_and_blank_text():
    assets = [{"assetId": "asset_a", "mediaPath": "assets/images/a.png"}]
    segments = SenseNovaU1Adapter._build_interleaved_segments("<image>Only a caption.", assets)
    assert segments == [
        {"type": "image", "assetId": "asset_a", "path": "assets/images/a.png"},
        {"type": "text", "text": "Only a caption."},
    ]


def test_interleave_segments_caps_images_to_available_assets():
    # More <image> markers than generated images: extra markers emit no image segment.
    assets = [{"assetId": "asset_a", "mediaPath": "assets/images/a.png"}]
    segments = SenseNovaU1Adapter._build_interleaved_segments("A<image>B<image>C", assets)
    assert segments == [
        {"type": "text", "text": "A"},
        {"type": "image", "assetId": "asset_a", "path": "assets/images/a.png"},
        {"type": "text", "text": "B"},
        {"type": "text", "text": "C"},
    ]


def test_write_interleaved_document_persists_document_and_image_assets(tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {"id": "job-1", "payload": {"projectId": "project-1", "model": "sensenova_u1_8b", "prompt": "guide"}}
    progress_results = []

    def progress(status, stage, value, message, result=None):
        if result is not None:
            progress_results.append(result)

    result = SenseNovaU1Adapter()._write_interleaved_document(
        project_path=project_path,
        request=image_request_from_job(job),
        job=job,
        project_id="project-1",
        model_id="sensenova_u1_8b",
        prompt="An illustrated guide to brewing tea",
        seed=7,
        generated_text="Boil water.<image>Steep.<image>Done.",
        images=[Image.new("RGB", (16, 16), (255, 0, 0)), Image.new("RGB", (16, 16), (0, 255, 0))],
        cancel_requested=lambda: False,
        progress=progress,
        raw_settings={"maxImages": 6, "resolution": "2048x1152"},
    )

    document_dir = project_path / "assets" / "documents"
    doc_jsons = [p for p in document_dir.glob("*.json") if not p.name.endswith(".sceneworks.json")]
    # The worker writes the document body; the Rust API builds the sidecar from
    # the document fact (story 1656 slice 4), so no sidecar lands here.
    assert len(doc_jsons) == 1
    assert not list(document_dir.glob("*.sceneworks.json"))

    document = json.loads(doc_jsons[0].read_text(encoding="utf-8"))
    assert [segment["type"] for segment in document["segments"]] == ["text", "image", "text", "image", "text"]
    assert document["model"] == "sensenova_u1_8b"
    assert document["jobId"] == "job-1"

    # The two generated images persist as ordinary image assets (Rust writes their
    # sidecars from facts); the worker still saves their PNG bytes, nested in the
    # generation-set subfolder.
    assert len(list((project_path / "assets" / "images").rglob("*.png"))) == 2
    assert len(result["imageAssetIds"]) == 2
    assert result["documentId"] == document["id"]
    assert result["segments"] == document["segments"]
    assert progress_results and progress_results[-1]["documentId"] == document["id"]

    # The worker emits flat facts: two image facts + one document fact; the Rust
    # API builds + indexes all three sidecars on completion.
    asset_writes = result["assetWrites"]
    assert len(asset_writes) == 3
    assert all(write["mimeType"] == "image/png" for write in asset_writes[:2])
    document_write = asset_writes[-1]
    assert document_write["type"] == "document"
    assert document_write["assetId"] == result["documentAssetId"]
    assert document_write["mediaPath"] == f"assets/documents/{document['id']}.json"
    assert document_write["mimeType"] == "application/json"
    assert document_write["mode"] == "interleave"
    assert document_write["imageCount"] == 2
    assert document_write["parents"] == result["imageAssetIds"]


def test_sensenova_resolution_for_snaps_to_buckets():
    # Snaps any request to the nearest SenseNova-U1 trained bucket by aspect ratio.
    assert sensenova_resolution_for(1024, 1024) == (2048, 2048)
    assert sensenova_resolution_for(1920, 1080) == (2720, 1536)
    assert sensenova_resolution_for(1080, 1920) == (1536, 2720)
    assert sensenova_resolution_for(1024, 768) == (2368, 1760)
    assert sensenova_resolution_for(768, 1024) == (1760, 2368)
    assert sensenova_resolution_for(1500, 1000) == (2496, 1664)


def test_image_request_allows_sensenova_wide_buckets():
    # The W/H clamp was raised (2048 -> 4096) so SenseNova's true trained buckets
    # (up to 3456) survive into the adapter, which snaps by the requested aspect
    # ratio. Truncating 2720x1536 to 2048x1536 would mis-snap 16:9 to a ~4:3 bucket.
    request = image_request_from_job({"payload": {"projectId": "p", "width": 2720, "height": 1536}})
    assert (request.width, request.height) == (2720, 1536)
    assert sensenova_resolution_for(request.width, request.height) == (2720, 1536)
    assert sensenova_resolution_for(2048, 1536) != (2720, 1536)


def test_image_request_clamps_above_new_ceiling():
    request = image_request_from_job({"payload": {"projectId": "p", "width": 9000, "height": 9000}})
    assert request.width == 4096
    assert request.height == 4096


def test_image_request_parses_optional_upscale_contract():
    default_request = image_request_from_job({"payload": {"projectId": "p"}})
    assert default_request.upscale.enabled is False
    assert default_request.upscale.factor == 2
    assert default_request.upscale.engine == "real-esrgan"

    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 4, "engine": "real-esrgan"},
            }
        }
    )
    assert request.upscale.enabled is True
    assert request.upscale.factor == 4
    assert request.upscale.engine == "real-esrgan"


def test_create_image_upscaler_rejects_unknown_engine():
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 2, "engine": "mystery-upscale"},
            }
        }
    )

    with pytest.raises(RuntimeError, match="Unsupported image upscale engine"):
        create_image_upscaler(request)


def test_create_image_upscaler_accepts_aurasr_engine():
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 4, "engine": "aura-sr"},
            }
        }
    )

    upscaler = create_image_upscaler(request, settings=SimpleNamespace(gpu_id="cpu"))

    assert isinstance(upscaler, AuraSrUpscaler)
    assert upscaler.id == "aura-sr"


def test_real_esrgan_resolves_manifest_weight_through_hf_cache(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    resolved_path = tmp_path / "RealESRGAN_x2plus.pth"
    resolved_path.write_bytes(b"weights")

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise RuntimeError("cache miss")
        return str(resolved_path)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    settings = SimpleNamespace(data_dir=tmp_path / "data", gpu_id="cpu")
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 2, "engine": "real-esrgan"},
                "modelManifestEntry": {
                    "resources": {
                        "imageUpscalers": {
                            "real-esrgan": {
                                "x2": {"repo": "example/upscalers", "file": "RealESRGAN_x2plus.pth"}
                            }
                        }
                    }
                },
            }
        }
    )

    path = RealEsrganUpscaler(settings=settings)._resolve_model_path(
        request,
        REAL_ESRGAN_MODEL_SPECS[2],
    )

    assert path == resolved_path
    assert calls[0]["repo_id"] == "example/upscalers"
    assert calls[0]["filename"] == "RealESRGAN_x2plus.pth"
    assert calls[0]["cache_dir"] == str(settings.data_dir / "cache" / "huggingface" / "hub")
    assert calls[0]["local_files_only"] is True
    assert calls[-1].get("local_files_only") is not True


def test_aurasr_resolves_manifest_snapshot_through_hf_cache(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "model.safetensors").write_bytes(b"weights")
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise RuntimeError("cache miss")
        return str(snapshot_dir)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    settings = SimpleNamespace(data_dir=tmp_path / "data", gpu_id="cpu")
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 4, "engine": "aura-sr"},
                "modelManifestEntry": {
                    "resources": {
                        "imageUpscalers": {
                            "aura-sr": {
                                "x4": {"repo": "example/aura-sr", "file": "model.safetensors"}
                            }
                        }
                    }
                },
            }
        }
    )

    path = AuraSrUpscaler(settings=settings)._resolve_model_path(request)

    assert path == snapshot_dir / "model.safetensors"
    assert calls[0]["repo_id"] == "example/aura-sr"
    assert calls[0]["allow_patterns"] == ["model.safetensors", "config.json", "LICENSE.md", "README.md"]
    assert calls[0]["cache_dir"] == str(settings.data_dir / "cache" / "huggingface" / "hub")
    assert calls[0]["local_files_only"] is True
    assert calls[-1].get("local_files_only") is not True


def test_aurasr_upscaler_rejects_non_4x_request(monkeypatch):
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 2, "engine": "aura-sr"},
            }
        }
    )

    with pytest.raises(RuntimeError, match="supports only 4x"):
        AuraSrUpscaler(settings=SimpleNamespace(gpu_id="cpu")).upscale(
            Image.new("RGB", (2, 2), "white"),
            request=request,
            cancel_requested=lambda: False,
        )


def test_image_asset_writer_retains_original_and_adds_upscaled_variant(monkeypatch, tmp_path):
    class FakeUpscaler:
        id = "real-esrgan"

        def upscale(self, image, *, request, cancel_requested):
            assert request.upscale.factor == 2
            assert cancel_requested() is False
            return image.resize((image.width * 2, image.height * 2))

    monkeypatch.setattr("scene_worker.image_adapters.create_image_upscaler", lambda *_args, **_kwargs: FakeUpscaler())
    project_path = tmp_path / "project"
    project_path.mkdir()
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "z_image_turbo",
            "count": 1,
            "width": 16,
            "height": 16,
            "upscale": {"enabled": True, "factor": 2, "engine": "real-esrgan"},
        },
    }

    result = ImageAssetWriter().write_incremental_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        image_count=1,
        image_at_index=lambda _index: Image.new("RGB", (16, 12), "navy"),
        adapter_id="z_image_diffusers",
        progress=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
        job_id=job["id"],
    )

    assert result["expectedCount"] == 2
    assert len(result["assetWrites"]) == 2

    original_write, upscaled_write = result["assetWrites"]
    assert (original_write["width"], original_write["height"]) == (16, 12)
    assert "upscale" not in original_write["rawAdapterSettings"]
    assert upscaled_write["sourceAssetId"] == original_write["assetId"]
    assert upscaled_write["parents"] == [original_write["assetId"]]
    assert upscaled_write["extra"] == {
        "isUpscaled": True,
        "upscaledFromAssetId": original_write["assetId"],
        "factor": 2,
        "engine": "real-esrgan",
    }
    assert (upscaled_write["width"], upscaled_write["height"]) == (32, 24)
    assert upscaled_write["rawAdapterSettings"]["upscale"] == {
        "enabled": True,
        "engine": "real-esrgan",
        "factor": 2,
        "sourceWidth": 16,
        "sourceHeight": 12,
        "width": 32,
        "height": 24,
    }
    with Image.open(project_path / original_write["mediaPath"]) as saved:
        assert saved.size == (16, 12)
    with Image.open(project_path / upscaled_write["mediaPath"]) as saved:
        assert saved.size == (32, 24)


def test_create_image_adapter_routes_sensenova_u1_fast():
    adapter = create_image_adapter({"payload": {"model": "sensenova_u1_8b_fast"}})
    assert adapter.__class__.__name__ == "SenseNovaU1Adapter"
    assert adapter.id == "sensenova_u1"


def test_sensenova_u1_fast_model_target_defaults():
    target = MODEL_TARGETS["sensenova_u1_8b_fast"]
    assert target["adapter"] == "sensenova_u1"
    assert target["family"] == "sensenova-u1"
    # 8-step distill LoRA: shares the base weights, cfg 1.0 / 8 steps.
    assert target["steps"] == 8
    assert target["guidanceScale"] == 1.0
    # Distilled editing (it2i) reuses the same variant + distill LoRA.
    assert target["supportsEdit"] is True
    assert target["repo"] == "sensenova/SenseNova-U1-8B-MoT"
    assert target["distillLora"] == {
        "repo": "sensenova/SenseNova-U1-8B-MoT-LoRAs",
        "file": "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors",
    }


def test_sensenova_u1_guidance_scale_defaults_from_model_target():
    request = image_request_from_job({"payload": {"projectId": "project_x", "prompt": "a cat"}})
    # The fast variant defaults to cfg 1.0; the base variant to cfg 4.0.
    assert SenseNovaU1Adapter._guidance_scale(request, MODEL_TARGETS["sensenova_u1_8b_fast"]) == 1.0
    assert SenseNovaU1Adapter._guidance_scale(request, MODEL_TARGETS["sensenova_u1_8b"]) == 4.0
    # An explicit request value overrides the per-model default.
    override = image_request_from_job(
        {"payload": {"projectId": "project_x", "prompt": "a cat", "advanced": {"guidanceScale": 2.5}}}
    )
    assert SenseNovaU1Adapter._guidance_scale(override, MODEL_TARGETS["sensenova_u1_8b_fast"]) == 2.5


def test_huggingface_repo_cache_path_stays_under_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    path = huggingface_repo_cache_path(r"..\outside/../../model")

    assert path is not None
    path.relative_to((tmp_path / "hub").resolve())
    assert path.name.startswith("models--")


def test_repo_cache_path_is_a_single_guarded_helper(monkeypatch, tmp_path):
    # The lora adapter previously had its own unguarded copy with a different
    # cache-root order. Every adapter must now resolve through the one guarded
    # helper, so a traversal repo id can't escape the cache root from any entry point.
    from scene_worker.hf_cache import huggingface_repo_cache_path as shared
    from scene_worker.lora_adapters import huggingface_repo_cache_path as lora_entry

    assert huggingface_repo_cache_path is shared
    assert lora_entry is shared

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    path = lora_entry(r"..\..\outside/model")
    assert path is not None
    path.relative_to((tmp_path / "hub").resolve())
    assert path.name.startswith("models--")


def test_image_asset_writer_reports_partial_result_assets(tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "z_image_turbo",
            "count": 2,
            "width": 16,
            "height": 16,
        },
    }
    progress_calls = []

    def progress(status, stage, value, message, result=None):
        progress_calls.append(
            {
                "status": status,
                "stage": stage,
                "value": value,
                "message": message,
                "result": result,
            }
        )

    result = ImageAssetWriter().write_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        images=[
            Image.new("RGB", (16, 16), (255, 0, 0)),
            Image.new("RGB", (16, 16), (0, 255, 0)),
        ],
        adapter_id="procedural_preview",
        progress=progress,
        cancel_requested=lambda: False,
        raw_settings={"preview": True},
    )

    result_progress = [call["result"] for call in progress_calls if call["result"]]
    # The worker reports flat facts now (story 1656); the Rust API builds the
    # sidecars and injects assets/assetIds. So the worker-side result streams the
    # growing assetWrites list, one more entry per image.
    assert [len(item["assetWrites"]) for item in result_progress] == [1, 2]
    assert result_progress[0]["expectedCount"] == 2
    assert result_progress[0]["generationSetId"] == result["generationSetId"]
    assert result_progress[0]["assetWrites"][0]["mediaPath"].startswith("assets/images/")
    assert [write["assetId"] for write in result_progress[1]["assetWrites"]] == [
        write["assetId"] for write in result["assetWrites"]
    ]
    assert result["expectedCount"] == 2


def test_image_asset_writer_persists_each_image_before_requesting_next(tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "z_image_turbo",
            "count": 2,
            "width": 16,
            "height": 16,
        },
    }

    def image_at_index(index):
        if index == 1:
            # The worker saves each PNG before requesting the next image (so a
            # multi-image batch streams). Sidecars are written by Rust now (story
            # 1656), so only the PNG lands on disk here — rglob into the
            # per-generation-set subfolder.
            assert len(list((project_path / "assets" / "images").rglob("*.png"))) == 1
        return Image.new("RGB", (16, 16), (255, 0, 0) if index == 0 else (0, 255, 0))

    result = ImageAssetWriter().write_incremental_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        image_count=2,
        image_at_index=image_at_index,
        adapter_id="z_image_diffusers",
        progress=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
    )

    assert len(result["assetWrites"]) == 2
    assert len(list((project_path / "assets" / "images").rglob("*.png"))) == 2


def test_image_asset_writer_does_not_clobber_identical_jobs(tmp_path):
    # Two jobs sharing the same date + model + prompt + image index must not
    # collide on disk: the per-generation-set subfolder keeps each job's PNG
    # distinct so the first asset's pixels are never overwritten by the second.
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "sensenova_u1_8b",
            "count": 1,
            "width": 16,
            "height": 16,
        },
    }

    def run(color):
        return ImageAssetWriter().write_incremental_outputs(
            request=image_request_from_job(job),
            project_path=project_path,
            image_count=1,
            image_at_index=lambda _index: Image.new("RGB", (16, 16), color),
            adapter_id="sensenova_u1",
            progress=lambda *_args, **_kwargs: None,
            cancel_requested=lambda: False,
            raw_settings={"realModelInference": True},
        )

    first = run((255, 0, 0))
    second = run((0, 255, 0))

    # Distinct generation sets, distinct asset ids, distinct files on disk.
    assert first["generationSetId"] != second["generationSetId"]
    assert [write["assetId"] for write in first["assetWrites"]] != [
        write["assetId"] for write in second["assetWrites"]
    ]
    pngs = list((project_path / "assets" / "images").rglob("*.png"))
    assert len(pngs) == 2

    # The first asset's recorded path still points at the first job's pixels;
    # the second job did not overwrite it.
    first_path = project_path / first["assetWrites"][0]["mediaPath"]
    second_path = project_path / second["assetWrites"][0]["mediaPath"]
    assert first_path.exists() and second_path.exists()
    assert first_path != second_path
    with Image.open(first_path) as handle:
        assert handle.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    with Image.open(second_path) as handle:
        assert handle.convert("RGB").getpixel((0, 0)) == (0, 255, 0)


def test_image_asset_writer_records_actual_output_dimensions(tmp_path):
    # The reported width/height (which Rust writes into the sidecar's file block)
    # must reflect the PNG actually saved (e.g. a model that snaps to a trained
    # bucket), not the requested size or the old min(request, 1280) cap.
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "sensenova_u1_8b",
            "count": 1,
            "width": 2720,
            "height": 1536,
        },
    }

    result = ImageAssetWriter().write_incremental_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        image_count=1,
        image_at_index=lambda _index: Image.new("RGB", (2720, 1536), "navy"),
        adapter_id="sensenova_u1",
        progress=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
    )

    write = result["assetWrites"][0]
    assert (write["width"], write["height"]) == (2720, 1536)


def test_image_asset_writer_batch_progress_is_monotonic(tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "z_image_turbo",
            "count": 4,
            "width": 16,
            "height": 16,
        },
    }
    progress_values = []

    def progress(_status, _stage, value, _message, result=None):
        progress_values.append(value)

    def image_at_index(index):
        progress("running", "generating", image_batch_progress(index, 4), f"Running image {index + 1} of 4.")
        return Image.new("RGB", (16, 16), (255, 0, 0))

    ImageAssetWriter().write_incremental_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        image_count=4,
        image_at_index=image_at_index,
        adapter_id="z_image_diffusers",
        progress=progress,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
    )

    assert progress_values == sorted(progress_values)


def test_friendly_failure_identifies_gpu_oom():
    message, error = friendly_failure("Image generation", RuntimeError("CUDA error: out of memory"))

    assert message == "Image generation failed because the GPU ran out of memory."
    assert "lower resolution" in error
    assert "Technical detail" in error


def test_friendly_failure_identifies_cpu_only_torch_worker():
    message, error = friendly_failure("Image generation", RuntimeError("CUDA-enabled PyTorch is not available."))

    assert message == "Image generation failed because the worker is missing CUDA-enabled PyTorch."
    assert "Rebuild the worker image" in error
    assert "Technical detail" in error


def test_friendly_failure_identifies_missing_model_files():
    message, error = friendly_failure("Image generation", RuntimeError("Repository not found: owner/model"))

    assert message == "Image generation failed because required model files were not available."
    assert "Model Manager" in error
    assert "Rust utility worker" in error
    assert "HF_TOKEN" in error


def test_friendly_failure_identifies_missing_diffusers_model_index():
    message, error = friendly_failure(
        "Video generation",
        RuntimeError(
            "404 Client Error. Entry Not Found for url: "
            "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/model_index.json."
        ),
    )

    assert message == "Video generation failed because required model files were not available."
    assert "model_index.json" in error
    assert "Technical detail" in error


def test_friendly_failure_identifies_ltx_frame_count_errors():
    message, error = friendly_failure("Video generation", RuntimeError("num_frames must be divisible by 8 + 1"))

    assert message == "Video generation failed because LTX requires a compatible frame count."
    assert "(frames - 1)" in error
    assert "Technical detail" in error


def test_friendly_failure_identifies_missing_peft_backend():
    message, error = friendly_failure(
        "Image generation",
        RuntimeError("LoRA style requires the PEFT backend for z_image_diffusers."),
    )

    assert message == "Image generation failed because the selected preset or LoRA needs PEFT support."
    assert "pip install -r apps/worker/requirements.txt" in error
    assert "docker compose build worker --no-cache" in error
    assert "Technical detail" in error


def test_friendly_failure_identifies_missing_sentencepiece_backend():
    message, error = friendly_failure(
        "Video generation",
        RuntimeError(
            "The component <class 'transformers.models.t5.tokenization_t5."
            "_LazyModule.__getattr__.<locals>.Placeholder'> of <class "
            "'diffusers.pipelines.ltx.pipeline_ltx_image2video.LTXImageToVideoPipeline'> "
            "cannot be loaded as it does not seem to have any of the loading methods defined."
        ),
    )

    assert message == "Video generation failed because the worker is missing a tokenizer backend."
    assert "tokenizer support libraries" in error
    assert "pip install -r apps/worker/requirements.txt" in error
    assert "docker compose build worker --no-cache" in error
    assert "Technical detail" in error


def test_friendly_failure_identifies_missing_protobuf_backend():
    message, error = friendly_failure(
        "Training captioning",
        RuntimeError("requires the protobuf library but it was not found in your environment"),
    )

    assert message == "Training captioning failed because the worker is missing a tokenizer backend."
    assert "tokenizer support libraries" in error
    assert "pip install -r apps/worker/requirements.txt" in error


def test_friendly_failure_identifies_disk_full_by_message():
    message, error = friendly_failure(
        "LoRA training", RuntimeError("OSError: [Errno 28] No space left on device")
    )

    assert message == "LoRA training failed because the disk ran out of space."
    assert "Free up disk space" in error
    assert "Technical detail" in error


def test_friendly_failure_identifies_disk_full_by_oserror_errno():
    message, error = friendly_failure(
        "LoRA training", OSError(28, "No space left on device")
    )

    assert message == "LoRA training failed because the disk ran out of space."
    assert "Free up disk space" in error


def test_is_cuda_oom_detects_oom_by_type_and_message():
    class OutOfMemoryError(RuntimeError):
        pass

    assert is_cuda_oom(OutOfMemoryError("allocator failed"))
    assert is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_cuda_oom(RuntimeError("some other failure"))


def test_joy_caption_prompt_builder_applies_length_and_name_options():
    options = JoyCaptionOptions(
        caption_type="Descriptive",
        caption_length="40",
        extra_options=["If there is a person/character in the image you must refer to them as {name}."],
        name_input="Mira",
    )

    prompt = build_joy_caption_prompt(options)

    assert "40 words or less" in prompt
    assert "Mira" in prompt


def test_caption_with_trigger_words_prepends_missing_tokens():
    caption = caption_with_trigger_words("studio portrait with soft light", ["miraStyle", "studio"])

    assert caption == "miraStyle, studio portrait with soft light"


def test_normalize_processor_resample_replaces_unsupported_lanczos():
    processor = SimpleNamespace(image_processor=SimpleNamespace(resample="lanczos"))

    normalize_processor_resample(processor)

    assert processor.image_processor.resample == JOY_CAPTION_RESAMPLE


def test_worker_check_reports_inference_sidecar_capabilities(monkeypatch):
    events = []
    monkeypatch.setattr("scene_worker.runtime.emit", events.append)
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.detector_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.tracker_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.segmenter_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.kps_extractor_backend_available", lambda: True)
    monkeypatch.setattr(
        "scene_worker.runtime.discover_gpu",
        lambda _gpu_id: {"id": "0", "name": "GPU 0", "capabilities": ["gpu"]},
    )

    run_check(SimpleNamespace(worker_id="worker-1", gpu_id="0"))

    assert events[0]["event"] == "worker_check"
    assert events[0]["jobTypes"] == [
        "image_generate",
        "image_edit",
        "video_generate",
        "video_extend",
        "video_bridge",
        "person_replace",
        "person_detect",
        "person_track",
        "pose_detect",
        # sc-4433: SCRFD 5-point landmark extraction (Key Point Library).
        "kps_extract",
        "lora_train",
        "training_caption",
        # sc-1635: VQA + interleave are advertised and dispatched, so the check
        # must report them too.
        "image_vqa",
        "image_interleave",
        # sc-2041: prompt refinement is dispatched on every Python worker.
        "prompt_refine",
        # sc-2431: standalone image upscale (Image Editor).
        "image_upscale",
        # sc-2438: standalone tile-ControlNet detail refine (Image Editor).
        "image_detail",
    ]
    assert events[0]["supportedJobTypes"] == events[0]["jobTypes"]


class _DryRunApi:
    """Records job progress posts; heartbeats are accepted and ignored. Job GETs
    report no cancellation so a real run completes unless a test says otherwise."""

    def __init__(self, cancel_requested=False):
        self.progress = []
        self._cancel_requested = cancel_requested

    def post(self, path, payload):
        if path.endswith("/heartbeat"):
            return {}
        if path.endswith("/progress"):
            self.progress.append(payload)
            return {"status": payload["status"], "stage": payload["stage"]}
        raise AssertionError(path)

    def get(self, _path):
        return {"cancelRequested": self._cancel_requested}


def _lora_train_job(plan):
    return {
        "id": "job-train-1",
        "type": "lora_train",
        "payload": {"dryRun": True, "plan": plan},
    }


def test_lora_train_dry_run_completes_with_plan_summary(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    image = tmp_path / "images" / "001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    api = _DryRunApi()
    plan = {
        "planVersion": 1,
        "dataset": {
            "datasetId": "ds_1",
            "datasetVersion": 2,
            "items": [{"imagePath": str(image), "caption": "auroraStyle portrait"}],
        },
        "target": {
            "targetId": "z_image_turbo_lora",
            "kernel": "z_image_lora",
            "baseModelPath": str(tmp_path / "uninstalled-model"),
        },
        "output": {
            "loraId": "lora_1",
            "outputDir": str(tmp_path / "loras" / "lora_1"),
            "fileName": "aurora.safetensors",
        },
    }

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="cpu"), _lora_train_job(plan))

    terminal = api.progress[-1]
    assert terminal["status"] == "completed"
    assert terminal["stage"] == "completed"
    result = terminal["result"]
    assert result["mode"] == "dry_run"
    assert result["validated"] is True
    assert result["datasetItemCount"] == 1
    assert result["loraId"] == "lora_1"
    assert result["fileName"] == "aurora.safetensors"
    # The base model is not installed yet; the dry run records that without failing.
    assert result["baseModelInstalled"] is False


def test_lora_train_dry_run_fails_cleanly_on_missing_dataset_image(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    api = _DryRunApi()
    plan = {
        "planVersion": 1,
        "dataset": {"items": [{"imagePath": str(tmp_path / "missing.png"), "caption": "x"}]},
        "target": {},
        "output": {},
    }

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="cpu"), _lora_train_job(plan))

    terminal = api.progress[-1]
    assert terminal["status"] == "failed"
    assert "missing" in terminal["error"].lower()


def test_lora_train_dry_run_fails_on_unsupported_plan_version(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    api = _DryRunApi()
    plan = {"planVersion": 999, "dataset": {"items": []}, "target": {}, "output": {}}

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="cpu"), _lora_train_job(plan))

    terminal = api.progress[-1]
    assert terminal["status"] == "failed"
    assert "version" in terminal["error"].lower()


def test_dry_run_summary_records_base_model_repo_and_install_state(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    plan = {
        "planVersion": 1,
        "dataset": {"datasetId": "ds_1", "datasetVersion": 4, "items": [{"imagePath": "x"}]},
        "target": {
            "targetId": "z_image_turbo_lora",
            "kernel": "z_image_lora",
            "baseModel": "z_image_turbo",
            "baseModelRepo": "Tongyi-MAI/Z-Image-Turbo",
            "baseModelPath": str(model_dir),
        },
        "output": {"loraId": "lora_1", "outputDir": str(tmp_path / "out"), "fileName": "a.safetensors"},
    }

    summary = dry_run_training_summary(plan, dry_run=True)

    assert summary["baseModelRepo"] == "Tongyi-MAI/Z-Image-Turbo"
    assert summary["baseModelInstalled"] is True
    assert summary["datasetVersion"] == 4


def test_validate_training_plan_rejects_bad_version_and_empty_dataset():
    with pytest.raises(ValueError, match="version"):
        validate_training_plan({"planVersion": 99, "dataset": {"items": [{"imagePath": "x"}]}})
    with pytest.raises(ValueError, match="no items"):
        validate_training_plan({"planVersion": SUPPORTED_TRAINING_PLAN_VERSION, "dataset": {"items": []}})


def test_bucket_resolution_floors_to_multiple_of_32():
    assert bucket_resolution(1024) == 1024
    assert bucket_resolution(1000) == 992
    assert bucket_resolution(20) == 32


def test_flow_matching_velocity_target_uses_negated_pipeline_sign():
    # The raw transformer output target is latents - noise, NOT noise - latents:
    # diffusers' ZImagePipeline negates the transformer output before the scheduler,
    # so the trained raw output is the negated flow velocity. Pin the sign so a
    # refactor can't silently flip the training direction.
    assert flow_matching_velocity_target(0.0, 1.0) == -1.0
    assert flow_matching_velocity_target(2.0, -1.0) == 3.0
    latents, noise = 0.7, 0.2
    assert flow_matching_velocity_target(latents, noise) == latents - noise
    assert flow_matching_velocity_target(latents, noise) == -(noise - latents)


def test_sample_training_timestep_accepts_ai_toolkit_shape_and_bias():
    torch = pytest.importorskip("torch")
    generator = torch.Generator("cpu").manual_seed(7)

    timestep = sample_training_timestep(
        torch,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
        timestep_type="sigmoid",
        timestep_bias="high_noise",
    )

    assert timestep.shape == (1,)
    assert float(timestep.item()) > 0.001
    assert float(timestep.item()) < 0.999


def test_seeded_sample_draws_on_generator_device_then_moves():
    """Regression for the MPS crash: a cpu ``torch.Generator`` cannot drive a
    ``torch.randn(..., device='mps')`` call. ``seeded_sample`` must draw on the
    generator's own device and move to the target when they differ, and pass the
    device straight through when they match. (``meta`` stands in for a non-cpu
    target so the routing is exercised without an actual GPU/MPS backend.)"""
    torch = pytest.importorskip("torch")
    generator = torch.Generator("cpu").manual_seed(11)

    seen_devices: list[str] = []

    def fake_fn(shape, *, generator, device, dtype):  # noqa: ARG001
        seen_devices.append(str(device))
        return torch.zeros(shape, dtype=dtype)

    # Matching device type → pass through, no move.
    out = seeded_sample(torch, fake_fn, (2,), generator=generator, device="cpu", dtype=torch.float32)
    assert seen_devices == ["cpu"]
    assert out.device.type == "cpu"

    # Mismatched target (cpu generator → non-cpu device) → draw on cpu, then move.
    seen_devices.clear()
    moved = seeded_sample(torch, fake_fn, (2,), generator=generator, device="meta", dtype=torch.float32)
    assert seen_devices == ["cpu"], "must generate on the generator's device, not the mismatched target"
    assert moved.device.type == "meta", "result must be moved to the requested device"


def test_build_optimizer_uses_prodigy_with_aitoolkit_lr_floor(monkeypatch):
    calls = {}

    class FakeProdigy:
        def __init__(self, params, **kwargs):
            calls["params"] = params
            calls["kwargs"] = kwargs

    def fake_import_module(name):
        if name == "torch":
            return SimpleNamespace(optim=SimpleNamespace())
        if name == "prodigyopt":
            return SimpleNamespace(Prodigy=FakeProdigy)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", fake_import_module)
    params = [object()]

    optimizer = build_optimizer("prodigyopt", params, 0.0001, 0.0001)

    assert isinstance(optimizer, FakeProdigy)
    assert calls == {"params": params, "kwargs": {"lr": 1.0, "eps": 1e-6, "weight_decay": 0.0001}}


def test_build_optimizer_uses_rose(monkeypatch):
    calls = {}

    class FakeRose:
        def __init__(self, params, **kwargs):
            calls["params"] = params
            calls["kwargs"] = kwargs

    def fake_import_module(name):
        if name == "torch":
            return SimpleNamespace(optim=SimpleNamespace())
        if name == "rose_opt":
            return SimpleNamespace(Rose=FakeRose)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", fake_import_module)
    params = [object()]

    optimizer = build_optimizer("rose", params, 0.0005, 0.01)

    assert isinstance(optimizer, FakeRose)
    # compute_dtype="fp32" is pinned to avoid Rose's fp64 default, which has no
    # MPS kernel on Apple Silicon.
    assert calls == {
        "params": params,
        "kwargs": {"lr": 0.0005, "weight_decay": 0.01, "compute_dtype": "fp32"},
    }


def _run_config_for_network(advanced=None):
    plan = {"config": {"rank": 8, "alpha": 8, "advanced": advanced or {}}}
    return read_run_config(plan)


def test_read_run_config_defaults_network_type_to_lora():
    config = _run_config_for_network()
    assert config.network_type == "lora"
    assert config.decompose_factor == -1


def test_read_run_config_parses_lokr_network_and_factor():
    config = _run_config_for_network({"networkType": "LoKr", "decomposeFactor": 8})
    # networkType is normalized to lowercase so the trainer's equality check is stable.
    assert config.network_type == "lokr"
    assert config.decompose_factor == 8


def test_build_peft_network_config_defaults_to_lora():
    fake_peft = SimpleNamespace(
        LoraConfig=lambda **kw: ("lora", kw),
        LoKrConfig=lambda **kw: ("lokr", kw),
    )
    config = _run_config_for_network({"loraTargetModules": ["to_q", "to_v"]})
    kind, kwargs = build_peft_network_config(fake_peft, config)
    assert kind == "lora"
    assert kwargs == {
        "r": config.rank,
        "lora_alpha": config.alpha,
        "init_lora_weights": "gaussian",
        "target_modules": ["to_q", "to_v"],
    }


def test_build_peft_network_config_builds_lokr_with_decompose_factor():
    fake_peft = SimpleNamespace(
        LoraConfig=lambda **kw: ("lora", kw),
        LoKrConfig=lambda **kw: ("lokr", kw),
    )
    config = _run_config_for_network(
        {"networkType": "lokr", "decomposeFactor": 16, "loraTargetModules": ["to_q"]}
    )
    kind, kwargs = build_peft_network_config(fake_peft, config)
    assert kind == "lokr"
    assert kwargs == {
        "r": config.rank,
        "alpha": config.alpha,
        "decompose_factor": 16,
        "init_weights": True,
        "target_modules": ["to_q"],
    }


def test_write_lokr_adapter_stamps_metadata_and_serializes_cpu_tensors(monkeypatch, tmp_path):
    captured = {}

    def fake_save_file(tensors, path, metadata=None):
        captured["tensors"] = tensors
        captured["path"] = path
        captured["metadata"] = metadata
        Path(path).write_bytes(b"")

    fake_module = SimpleNamespace(save_file=fake_save_file)
    # Inject the parent package too so the function's `from safetensors.torch
    # import save_file` never touches the filesystem regardless of install state.
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(torch=fake_module))
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_module)

    class FakeTensor:
        def __init__(self):
            self.moved = []

        def detach(self):
            self.moved.append("detach")
            return self

        def cpu(self):
            self.moved.append("cpu")
            return self

        def contiguous(self):
            self.moved.append("contiguous")
            return self

    state = {"blk.lokr_w1": FakeTensor(), "blk.lokr_w2": FakeTensor()}
    path = write_lokr_adapter(
        state,
        str(tmp_path),
        "adapter.safetensors",
        rank=8,
        alpha=16,
        decompose_factor=8,
        target_modules=["to_q", "to_v"],
    )

    assert path == str(tmp_path / "adapter.safetensors")
    # Routing + reconstruction metadata: the inference loader (epic 2193) reads
    # networkType to route and rank/alpha/decomposeFactor/targetModules to rebuild
    # the matching LoKrConfig for injection.
    assert captured["metadata"] == {
        "format": "pt",
        "networkType": "lokr",
        "rank": "8",
        "alpha": "16",
        "decomposeFactor": "8",
        "targetModules": '["to_q", "to_v"]',
    }
    assert set(captured["tensors"]) == {"blk.lokr_w1", "blk.lokr_w2"}
    assert captured["tensors"]["blk.lokr_w1"].moved == ["detach", "cpu", "contiguous"]


def test_lr_schedule_updates_converts_microsteps_to_optimizer_updates():
    # accum=1 -> one optimizer update per micro-step; warmup passes through.
    assert lr_schedule_updates(1000, 1, 0) == (1000, 0)
    assert lr_schedule_updates(1000, 1, 100) == (1000, 100)
    # Gradient accumulation makes optimizer updates less frequent (ceil division),
    # and the warmup step count is converted the same way.
    assert lr_schedule_updates(1000, 4, 40) == (250, 10)
    assert lr_schedule_updates(10, 4, 0) == (3, 0)  # ceil(10 / 4)
    # Warmup is clamped strictly below the total so the body always decays.
    assert lr_schedule_updates(10, 1, 50) == (10, 9)


def test_normalize_lr_scheduler_accepts_supported_and_rejects_unknown():
    assert normalize_lr_scheduler(None) == "constant"
    assert normalize_lr_scheduler(" Cosine ") == "cosine"
    assert normalize_lr_scheduler("LINEAR") == "linear"
    for name in SUPPORTED_LR_SCHEDULERS:
        assert normalize_lr_scheduler(name) == name
    with pytest.raises(TrainingKernelError) as excinfo:
        normalize_lr_scheduler("warmup_cosine")
    assert "Unsupported lrScheduler" in str(excinfo.value)


def test_lr_decay_multiplier_curves():
    # Constant holds the base LR for the whole run.
    assert lr_decay_multiplier("constant", 0, 10, 0) == 1.0
    assert lr_decay_multiplier("constant", 5, 10, 0) == 1.0
    # Linear decays 1 -> 0 across the run.
    assert lr_decay_multiplier("linear", 0, 10, 0) == pytest.approx(1.0)
    assert lr_decay_multiplier("linear", 5, 10, 0) == pytest.approx(0.5)
    assert lr_decay_multiplier("linear", 10, 10, 0) == pytest.approx(0.0)
    # Cosine: 1 at the start, 0.5 at the midpoint, 0 at the end.
    assert lr_decay_multiplier("cosine", 0, 10, 0) == pytest.approx(1.0)
    assert lr_decay_multiplier("cosine", 5, 10, 0) == pytest.approx(0.5)
    assert lr_decay_multiplier("cosine", 10, 10, 0) == pytest.approx(0.0)
    # A linear warmup ramps to 1.0 without a dead zero-LR first step, then the
    # body schedule runs from its start (progress 0 -> multiplier 1.0).
    assert lr_decay_multiplier("cosine", 0, 10, 4) == pytest.approx(1 / 5)
    assert lr_decay_multiplier("cosine", 3, 10, 4) == pytest.approx(4 / 5)
    assert lr_decay_multiplier("cosine", 4, 10, 4) == pytest.approx(1.0)
    # Cosine and linear are different curves off the midpoint.
    assert lr_decay_multiplier("cosine", 3, 10, 0) != pytest.approx(
        lr_decay_multiplier("linear", 3, 10, 0)
    )


def test_build_lr_scheduler_rejects_unknown_before_touching_torch():
    # The name is validated before the optimizer/torch is used, so a dummy torch
    # is never dereferenced for an unknown scheduler.
    with pytest.raises(TrainingKernelError):
        build_lr_scheduler(
            SimpleNamespace(), SimpleNamespace(), "exotic", total_updates=5, warmup_updates=0
        )


def test_build_lr_scheduler_constant_is_fixed_and_cosine_linear_decay():
    torch = pytest.importorskip("torch")

    def make_optimizer():
        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([param], lr=0.1)

    # Plain constant (no warmup) returns no scheduler: the LR stays exactly fixed,
    # matching every pre-scheduler training run.
    optimizer = make_optimizer()
    assert (
        build_lr_scheduler(torch, optimizer, "constant", total_updates=10, warmup_updates=0)
        is None
    )

    def run(name):
        optimizer = make_optimizer()
        scheduler = build_lr_scheduler(torch, optimizer, name, total_updates=10, warmup_updates=0)
        assert scheduler is not None
        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        return lrs

    cosine = run("cosine")
    linear = run("linear")

    # Both start at the base LR, decay monotonically, and reach ~0 at the end.
    for lrs in (cosine, linear):
        assert lrs[0] == pytest.approx(0.1)
        assert all(later <= earlier + 1e-12 for earlier, later in zip(lrs, lrs[1:]))
        assert lrs[-1] == pytest.approx(0.0, abs=1e-7)
        assert lrs[5] < lrs[0]
    # Cosine and linear coincide at the exact midpoint (both 0.5) but differ
    # off-center, so compare away from it.
    assert cosine[3] != pytest.approx(linear[3])


def test_build_lr_scheduler_applies_linear_warmup_then_holds_for_constant():
    torch = pytest.importorskip("torch")
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)

    # Constant + warmup still builds a scheduler that ramps the LR in, then holds.
    scheduler = build_lr_scheduler(
        torch, optimizer, "constant", total_updates=10, warmup_updates=4
    )
    assert scheduler is not None

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Linear ramp 0.02 -> 0.04 -> 0.06 -> 0.08 -> 0.10, then held at the base LR.
    assert lrs[0] == pytest.approx(0.02)
    assert lrs[1] > lrs[0]
    assert lrs[4] == pytest.approx(0.1)
    assert lrs[5] == pytest.approx(0.1)


def test_read_run_config_parses_lr_scheduler_and_warmup():
    config = read_run_config(
        {"config": {"steps": 1200, "advanced": {"lrScheduler": "cosine", "lrWarmupSteps": 100}}}
    )
    assert config.lr_scheduler == "cosine"
    assert config.lr_warmup_steps == 100


def test_read_run_config_defaults_lr_scheduler_to_constant():
    config = read_run_config({"config": {}})
    assert config.lr_scheduler == "constant"
    assert config.lr_warmup_steps == 0


def test_read_run_config_defaults_weight_noise_sigma_to_zero():
    config = read_run_config({"config": {}})
    assert config.weight_noise_sigma == 0.0


def test_read_run_config_parses_weight_noise_sigma():
    config = read_run_config(
        {"config": {"advanced": {"weightNoiseSigma": 0.00125}}}
    )
    assert config.weight_noise_sigma == pytest.approx(0.00125)


def test_read_run_config_clamps_negative_weight_noise_sigma_to_zero():
    config = read_run_config(
        {"config": {"advanced": {"weightNoiseSigma": -0.01}}}
    )
    assert config.weight_noise_sigma == 0.0


def test_apply_weight_noise_is_no_op_when_sigma_is_zero():
    torch = pytest.importorskip("torch")

    param = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.SGD([param], lr=0.0)
    apply_weight_noise(torch, optimizer, 0.0)
    assert torch.equal(param.detach(), torch.zeros(4))


def test_apply_weight_noise_perturbs_with_expected_magnitude():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.zeros(4096))
    optimizer = torch.optim.SGD([param], lr=0.0)
    sigma = 0.01
    apply_weight_noise(torch, optimizer, sigma)
    # Population std of N(0, sigma) ~= sigma; allow a generous bound for 4096 samples.
    std = float(param.detach().std())
    assert std == pytest.approx(sigma, rel=0.1)
    # Mean should be near zero — perturbation is centered.
    assert abs(float(param.detach().mean())) < sigma


def test_apply_weight_noise_skips_frozen_params():
    torch = pytest.importorskip("torch")

    trainable = torch.nn.Parameter(torch.zeros(8))
    frozen = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
    optimizer = torch.optim.SGD([{"params": [trainable, frozen]}], lr=0.0)
    apply_weight_noise(torch, optimizer, 0.01)
    assert torch.equal(frozen.detach(), torch.zeros(8))
    assert not torch.equal(trainable.detach(), torch.zeros(8))


def test_z_image_lora_backend_activates_default_adapter():
    class FakeTransformer:
        def __init__(self):
            self.adapter_name = None

        def set_adapter(self, name):
            self.adapter_name = name

    transformer = FakeTransformer()

    _ZImageLoraBackend()._activate_lora_adapter(transformer)

    assert transformer.adapter_name == "default"


def test_read_run_config_parses_training_adapter():
    config = read_run_config(
        {
            "config": {
                "advanced": {
                    "trainingAdapterRepo": "ostris/zimage_turbo_training_adapter",
                    "trainingAdapterVersion": "v2-default",
                }
            }
        }
    )

    assert config.training_adapter_repo == "ostris/zimage_turbo_training_adapter"
    assert config.training_adapter_version == "v2-default"


def test_read_run_config_training_adapter_absent_is_none():
    config = read_run_config({"config": {"advanced": {}}})

    assert config.training_adapter_repo is None
    assert config.training_adapter_version is None


def test_training_adapter_weight_name_maps_versions():
    assert training_adapter_weight_name("v1") == "zimage_turbo_training_adapter_v1.safetensors"
    assert training_adapter_weight_name("v1-default") == "zimage_turbo_training_adapter_v1.safetensors"
    assert training_adapter_weight_name("v2-default") == "zimage_turbo_training_adapter_v2.safetensors"
    # Unknown / empty defaults to v2 (the SceneWorks preset default).
    assert training_adapter_weight_name(None) == "zimage_turbo_training_adapter_v2.safetensors"
    assert training_adapter_weight_name("") == "zimage_turbo_training_adapter_v2.safetensors"


def test_resolve_training_adapter_source_prefers_cached_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    repo = "ostris/zimage_turbo_training_adapter"
    repo_root = tmp_path / "hub" / "models--ostris--zimage_turbo_training_adapter"
    snapshot = repo_root / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir(parents=True)
    (repo_root / "refs" / "main").write_text("deadbeef", encoding="utf-8")
    weight_file = snapshot / "zimage_turbo_training_adapter_v2.safetensors"
    weight_file.write_bytes(b"weights")

    load_target, weight_name = resolve_training_adapter_source(repo, "v2-default")

    assert load_target == str(weight_file)
    assert weight_name == "zimage_turbo_training_adapter_v2.safetensors"


def test_resolve_training_adapter_source_falls_back_to_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    repo = "ostris/zimage_turbo_training_adapter"

    load_target, weight_name = resolve_training_adapter_source(repo, "v1")

    assert load_target == repo
    assert weight_name == "zimage_turbo_training_adapter_v1.safetensors"


def test_apply_training_adapter_fuses_and_unloads(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent-cache")
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    calls = []

    class FakePipe:
        def load_lora_weights(self, target, *, weight_name=None, adapter_name=None):
            calls.append(("load", target, weight_name, adapter_name))

        def fuse_lora(self):
            calls.append(("fuse",))

        def unload_lora_weights(self):
            calls.append(("unload",))

    config = read_run_config(
        {
            "config": {
                "advanced": {
                    "trainingAdapterRepo": "ostris/zimage_turbo_training_adapter",
                    "trainingAdapterVersion": "v2-default",
                }
            }
        }
    )

    weight_name = _ZImageLoraBackend()._apply_training_adapter(
        FakePipe(), config, lambda *args, **kwargs: None
    )

    assert weight_name == "zimage_turbo_training_adapter_v2.safetensors"
    assert [call[0] for call in calls] == ["load", "fuse", "unload"]
    load_call = calls[0]
    assert load_call[1] == "ostris/zimage_turbo_training_adapter"
    assert load_call[2] == "zimage_turbo_training_adapter_v2.safetensors"
    assert load_call[3] == "dedistill"


def test_apply_training_adapter_noop_without_repo():
    calls = []

    class FakePipe:
        def load_lora_weights(self, *args, **kwargs):
            calls.append("load")

        def fuse_lora(self):
            calls.append("fuse")

    config = read_run_config({"config": {"advanced": {}}})

    result = _ZImageLoraBackend()._apply_training_adapter(
        FakePipe(), config, lambda *args, **kwargs: None
    )

    assert result is None
    assert calls == []


def test_apply_training_adapter_raises_on_failure(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent-cache")
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    class FakePipe:
        def load_lora_weights(self, *args, **kwargs):
            raise RuntimeError("no such adapter")

        def fuse_lora(self):
            pass

    config = read_run_config(
        {
            "config": {
                "advanced": {
                    "trainingAdapterRepo": "ostris/zimage_turbo_training_adapter",
                }
            }
        }
    )

    with pytest.raises(TrainingKernelError, match="de-distill"):
        _ZImageLoraBackend()._apply_training_adapter(
            FakePipe(), config, lambda *args, **kwargs: None
        )


def test_read_run_config_defaults_lora_target_modules_and_parses_advanced():
    config = read_run_config(
        {
            "config": {
                "rank": 8,
                "alpha": 12,
                "learningRate": 0.0003,
                "steps": 500,
                "saveEvery": 100,
                "optimizer": "adamw8bit",
                "advanced": {
                    "mixedPrecision": "bf16",
                    "weightDecay": 0.0001,
                    "timestepType": "sigmoid",
                    "timestepBias": "high_noise",
                    "lossType": "mse",
                    "gradientCheckpointing": True,
                },
            }
        }
    )

    assert config.rank == 8
    assert config.alpha == 12
    assert config.steps == 500
    assert config.save_every == 100
    assert config.mixed_precision == "bf16"
    assert config.weight_decay == 0.0001
    assert config.timestep_type == "sigmoid"
    assert config.timestep_bias == "high_noise"
    assert config.loss_type == "mse"
    assert config.gradient_checkpointing is True
    assert config.lora_target_modules == ["to_q", "to_k", "to_v", "to_out.0"]
    assert config.sample_steps == 9
    assert config.sample_guidance_scale == 0.0


def test_read_run_config_parses_sample_render_settings():
    config = read_run_config(
        {
            "config": {
                "advanced": {
                    "sampleEvery": 50,
                    "sampleSteps": 12,
                    "sampleGuidanceScale": 1.25,
                    "samplePrompts": ["miraStyle portrait"],
                }
            }
        }
    )

    assert config.sample_every == 50
    assert config.sample_steps == 12
    assert config.sample_guidance_scale == 1.25
    assert config.sample_prompts == ["miraStyle portrait"]


def test_read_run_config_defaults_z_image_samples_to_turbo_guidance():
    config = read_run_config(
        {
            "target": {"kernel": "z_image_lora", "baseModel": "z_image_turbo"},
            "config": {"advanced": {"samplePrompts": ["miraStyle portrait"]}},
        }
    )

    assert config.sample_guidance_scale == 1.0


def test_z_image_lora_backend_generates_samples_with_turbo_guidance(tmp_path):
    calls = []

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeGenerator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakeTorch:
        def no_grad(self):
            return FakeNoGrad()

        def Generator(self, device):
            return FakeGenerator(device)

    class FakeImage:
        def convert(self, mode):
            assert mode == "RGB"
            return self

        def save(self, path):
            Path(path).write_bytes(b"png")

    class FakePipe:
        def __call__(
            self,
            *,
            prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            generator,
        ):
            calls.append(
                {
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": generator.seed,
                }
            )
            return SimpleNamespace(images=[FakeImage()])

    class FakeTransformer:
        training = True

        def set_adapter(self, _name):
            pass

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    backend = _ZImageLoraBackend()
    backend._torch = FakeTorch()
    backend._pipeline = FakePipe()
    backend._transformer = FakeTransformer()
    backend._device = "cpu"
    config = read_run_config(
        {
            "config": {
                "seed": 7,
                "advanced": {
                    "sampleSteps": 11,
                    "samplePrompts": ["miraStyle portrait"],
                },
            },
            "target": {"kernel": "z_image_lora", "baseModel": "z_image_turbo"},
            "output": {"triggerWords": ["miraStyle"]},
        }
    )

    samples = backend.generate_samples(
        step=4,
        prompts=config.sample_prompts,
        output_dir=str(tmp_path),
        file_name="mira.safetensors",
        plan={"dataset": {"rootPath": str(tmp_path / "training" / "datasets" / "ds_1")}},
        config=config,
    )

    assert calls[0]["num_inference_steps"] == 11
    assert calls[0]["guidance_scale"] == 1.0
    assert calls[0]["seed"] == 11
    assert samples[0]["sampleSource"] == "live_adapter"
    assert samples[0]["numInferenceSteps"] == 11
    assert samples[0]["guidanceScale"] == 1.0


def test_create_training_kernel_resolves_known_and_rejects_unknown():
    assert isinstance(create_training_kernel("z_image_lora"), ZImageLoraTrainer)
    assert isinstance(create_training_kernel("sdxl_lora"), SdxlLoraTrainer)
    assert isinstance(create_training_kernel("kolors_lora"), KolorsLoraTrainer)
    assert isinstance(create_training_kernel("wan_lora"), WanLoraTrainer)
    assert isinstance(create_training_kernel("wan_moe_lora"), WanMoeLoraTrainer)
    assert isinstance(create_training_kernel("lens_lora"), LensLoraTrainer)
    with pytest.raises(TrainingKernelError, match="No training kernel"):
        create_training_kernel("not_a_kernel")


def test_kolors_lora_trainer_reuses_sdxl_backend_with_kolors_seams():
    # KolorsLoraTrainer (epic 1929) is a thin extension of the generic SDXL-UNet
    # trainer: same orchestration + SDXL training loop, swapping only the pipeline
    # class + ChatGLM3 prompt encoder. LoKr is inherited from the SDXL backend.
    trainer = create_training_kernel("kolors_lora")
    assert isinstance(trainer, SdxlLoraTrainer)
    assert trainer.kernel_id == "kolors_lora"
    backend = trainer._create_backend()
    assert isinstance(backend, _KolorsLoraBackend)
    assert isinstance(backend, _SdxlLoraBackend)  # inherits the LoKr-wired save/load
    assert backend.kernel_id == "kolors_lora"
    assert backend.pipeline_class_name == "KolorsPipeline"


def test_kolors_encode_prompt_uses_chatglm_seam():
    # The only forward-pass seam: Kolors uses a single ChatGLM3 encoder (no SDXL
    # prompt_2) at GLM sequence length 256; the 4-tuple return order matches SDXL.
    captured: dict[str, object] = {}

    def fake_encode_prompt(**kwargs):
        captured.update(kwargs)
        return ("PROMPT_EMBEDS", "NEG", "POOLED", "NEG_POOLED")

    pipe = SimpleNamespace(encode_prompt=fake_encode_prompt)
    backend = _KolorsLoraBackend()
    prompt_embeds, pooled = backend._encode_prompt(pipe, "a calico cat", "cpu")

    assert prompt_embeds == "PROMPT_EMBEDS"
    assert pooled == "POOLED"
    assert captured["prompt"] == "a calico cat"
    assert captured["max_sequence_length"] == 256
    assert captured["do_classifier_free_guidance"] is False
    # Kolors has no second CLIP encoder — the SDXL `prompt_2` arg must be absent.
    assert "prompt_2" not in captured


def test_kolors_backend_releases_chatglm_encoder_when_not_sampling(monkeypatch):
    # ChatGLM3-6B is only needed to cache prompt embeddings. With no live sampling
    # the encoder is released after caching (Mac memory envelope, epic 1929); with
    # sampling on it's retained because generate_samples re-encodes prompts.
    monkeypatch.setattr(_SdxlLoraBackend, "prepare_dataset", lambda self, **kw: {"itemCount": 0})
    monkeypatch.setattr("scene_worker.training_adapters.empty_torch_cache", lambda _torch: None)

    def run(sample_every):
        backend = _KolorsLoraBackend()
        encoder = object()
        backend._pipeline = SimpleNamespace(text_encoder=encoder, text_encoder_2=None)
        backend.prepare_dataset(
            items=[],
            config=SimpleNamespace(sample_every=sample_every),
            progress=lambda *a, **k: None,
            cancel_requested=lambda: False,
        )
        return backend._pipeline.text_encoder, encoder

    released, _ = run(0)
    assert released is None  # released when not sampling
    retained, encoder = run(250)
    assert retained is encoder  # kept for live sampling


def test_wan_lora_trainer_reuses_zimage_orchestration_with_wan_backend():
    # WanLoraTrainer subclasses ZImageLoraTrainer (shared staged orchestration)
    # and only swaps the kernel id + the torch Wan backend it builds. The 14B MoE
    # trainer (sc-1953) extends this for the two-expert case.
    trainer = create_training_kernel("wan_lora")
    assert isinstance(trainer, ZImageLoraTrainer)
    assert trainer.kernel_id == "wan_lora"
    backend = trainer._create_backend()
    assert isinstance(backend, _WanLoraBackend)
    assert backend.kernel_id == "wan_lora"
    # Implements the full TrainingBackend protocol.
    for method in (
        "load",
        "prepare_dataset",
        "train_step",
        "save_checkpoint",
        "generate_samples",
        "save_final",
        "cleanup",
        "loaded_models",
    ):
        assert callable(getattr(backend, method)), method


def test_wan_lora_backend_read_run_config_uses_wan_target_modules():
    # The Rust wan_lora target declares the Wan transformer attention modules; the
    # kernel reads them straight from the plan's advanced config.
    plan = {
        "config": {
            "rank": 32,
            "steps": 1500,
            "advanced": {"loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"]},
        }
    }
    config = read_run_config(plan)
    assert list(config.lora_target_modules) == ["to_q", "to_k", "to_v", "to_out.0"]


def test_wan_moe_lora_trainer_extends_wan_backend():
    # WanMoeLoraTrainer subclasses the dense Wan trainer's backend for the A14B
    # two-expert case; it shares the orchestration and only swaps the backend.
    trainer = create_training_kernel("wan_moe_lora")
    assert isinstance(trainer, ZImageLoraTrainer)
    assert trainer.kernel_id == "wan_moe_lora"
    backend = trainer._create_backend()
    assert isinstance(backend, _WanMoeLoraBackend)
    assert isinstance(backend, _WanLoraBackend)
    assert backend.kernel_id == "wan_moe_lora"
    for method in ("load", "prepare_dataset", "train_step", "save_final", "cleanup"):
        assert callable(getattr(backend, method)), method


def test_wan_moe_lora_backend_parses_gguf_quant_spec_and_boundary():
    backend = _WanMoeLoraBackend()
    # Default boundary (A14B = 0.875) before any load.
    assert backend._boundary == 0.875
    # A complete gguf baseQuantization advanced block parses into an expert spec.
    gguf = read_run_config(
        {
            "config": {
                "advanced": {
                    "baseQuantization": {
                        "format": "gguf",
                        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
                        "highNoiseFile": "HighNoise/hi.gguf",
                        "lowNoiseFile": "LowNoise/lo.gguf",
                    }
                }
            }
        }
    )
    spec = backend._quant_spec(gguf)
    assert spec == {
        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
        "highNoiseFile": "HighNoise/hi.gguf",
        "lowNoiseFile": "LowNoise/lo.gguf",
    }
    # No quant block -> bf16 path (None); an incomplete block is ignored.
    assert backend._quant_spec(read_run_config({"config": {"advanced": {}}})) is None
    incomplete = read_run_config(
        {"config": {"advanced": {"baseQuantization": {"format": "gguf", "repo": "R"}}}}
    )
    assert backend._quant_spec(incomplete) is None


def test_sdxl_lora_trainer_reuses_zimage_orchestration_with_sdxl_backend():
    # SdxlLoraTrainer subclasses ZImageLoraTrainer (shared staged orchestration)
    # and only swaps the kernel id + the backend it builds.
    trainer = create_training_kernel("sdxl_lora")
    assert isinstance(trainer, ZImageLoraTrainer)
    assert trainer.kernel_id == "sdxl_lora"
    backend = trainer._create_backend()
    assert isinstance(backend, _SdxlLoraBackend)
    # Extension seams epic 1929 (Kolors) overrides: the pipeline class + the
    # prompt encoder. Everything else is shared.
    assert backend.kernel_id == "sdxl_lora"
    assert backend.pipeline_class_name == "StableDiffusionXLPipeline"
    assert backend.load_variant == "fp16"


def test_sdxl_lora_backend_read_run_config_uses_sdxl_target_modules():
    # The Rust sdxl_lora target declares the SDXL UNet attention modules; the
    # kernel reads them straight from the plan's advanced config.
    plan = {
        "config": {
            "rank": 16,
            "steps": 1500,
            "advanced": {"loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"]},
        }
    }
    config = read_run_config(plan)
    assert list(config.lora_target_modules) == ["to_q", "to_k", "to_v", "to_out.0"]


class _FakeLensSidecarPopen:
    """Stand-in for the Lens training sidecar process: on construction it reads
    the spec the driver wrote, emits a realistic progress.jsonl, writes the
    output adapter + result.json, and exits 0 — so ``LensLoraTrainer``'s
    subprocess orchestration is testable without the lens venv."""

    def __init__(self, cmd, env=None, stdout=None, stderr=None):
        spec = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        steps = int(spec["config"]["steps"])
        out_dir = Path(spec["outputDir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / spec["fileName"]
        output_path.write_bytes(b"lora")
        events = [
            {"event": "stage", "stage": "loading_model", "message": "loading"},
            {"event": "stage", "stage": "caching_latents", "message": "caching"},
            {"event": "cache", "done": 1, "total": 1},
            {"event": "stage", "stage": "training", "message": "training"},
            {"event": "step", "step": steps, "total": steps, "loss": 0.25},
            {"event": "saved", "path": str(output_path)},
        ]
        with Path(spec["progressPath"]).open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event) + "\n")
        Path(spec["resultPath"]).write_text(
            json.dumps(
                {
                    "outputPath": str(output_path),
                    "fileName": spec["fileName"],
                    "stepsCompleted": steps,
                    "checkpoints": [],
                    "trainingSamples": [],
                    "rank": spec["config"]["rank"],
                    "alpha": spec["config"]["alpha"],
                    "resolution": spec["config"]["resolution"],
                    "baseModelSource": spec["source"],
                }
            ),
            encoding="utf-8",
        )
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


def _lens_train_plan(tmp_path, *, steps=4):
    image = tmp_path / "images" / "000.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    return {
        "planVersion": 1,
        "dataset": {
            "datasetId": "ds_1",
            "datasetVersion": 1,
            "items": [{"imagePath": str(image), "caption": "auroraStyle"}],
        },
        "target": {
            "targetId": "lens_turbo_lora",
            "kernel": "lens_lora",
            "baseModel": "lens",
            "baseModelRepo": "microsoft/Lens",
            "baseModelPath": str(tmp_path / "absent"),
        },
        "config": {
            "rank": 16,
            "alpha": 16,
            "learningRate": 0.0001,
            "steps": steps,
            "batchSize": 1,
            "gradientAccumulation": 1,
            "resolution": 1024,
            "saveEvery": 0,
            "seed": 42,
            "optimizer": "adamw8bit",
            "advanced": {
                "lrScheduler": "constant",
                "loraTargetModules": ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"],
            },
        },
        "output": {
            "loraId": "lora_1",
            "outputDir": str(tmp_path / "loras" / "lora_1"),
            "fileName": "aurora.safetensors",
            "format": "safetensors",
            "triggerWords": ["auroraStyle"],
        },
    }


def test_lens_trainer_drives_sidecar_and_shapes_result(tmp_path, monkeypatch):
    import subprocess as _subprocess

    plan = _lens_train_plan(tmp_path, steps=4)
    monkeypatch.setattr(LensLoraTrainer, "_sidecar_available", lambda self: True)
    monkeypatch.setattr(_subprocess, "Popen", _FakeLensSidecarPopen)

    events = []
    result = LensLoraTrainer().train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda status, stage, value, message, result=None: events.append((status, stage)),
        cancel_requested=lambda: False,
    )

    stages = {stage for _status, stage in events}
    statuses = {status for status, _stage in events}
    # The driver maps the sidecar's JSONL events onto valid JobStatus bands; in
    # particular caching runs under "running" (not the invalid "caching").
    assert {"caching_latents", "training", "saving"}.issubset(stages)
    assert statuses <= _VALID_JOB_STATUSES
    assert result["mode"] == "train"
    assert result["kernel"] == "lens_lora"
    assert result["stepsCompleted"] == 4
    assert result["outputPath"] == os.path.join(plan["output"]["outputDir"], "aurora.safetensors")
    assert result["baseModelSource"] == "microsoft/Lens"
    assert result["triggerWords"] == ["auroraStyle"]
    assert os.path.exists(result["outputPath"])


def test_lens_trainer_requires_sidecar(tmp_path, monkeypatch):
    plan = _lens_train_plan(tmp_path)
    monkeypatch.setattr(LensLoraTrainer, "_sidecar_available", lambda self: False)
    with pytest.raises(TrainingKernelError, match="Lens sidecar venv"):
        LensLoraTrainer().train(
            settings=SimpleNamespace(worker_id="w", gpu_id="0"),
            plan=plan,
            progress=lambda *args, **kwargs: None,
            cancel_requested=lambda: False,
        )


def test_lens_device_hint_delegates_to_select_torch_device(monkeypatch):
    # On a real worker the driver has torch in the main venv, so the sidecar
    # device hint must come from select_torch_device — picking "mps" on Apple
    # Silicon exactly like the Lens inference adapter, not a hardcoded "cuda".
    captured = {}

    def _fake_import(name):
        captured["imported"] = name
        return SimpleNamespace(__name__="torch")

    def _fake_select(torch, gpu_id):
        captured["gpu_id"] = gpu_id
        return "mps"

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", _fake_import)
    monkeypatch.setattr("scene_worker.training_adapters.select_torch_device", _fake_select)

    device = LensLoraTrainer._device_hint(SimpleNamespace(gpu_id="0"))
    assert device == "mps"
    assert captured["imported"] == "torch"
    assert captured["gpu_id"] == "0"


def test_lens_device_hint_falls_back_to_mps_without_torch(monkeypatch):
    # If torch is somehow unimportable in the main venv, the hint still resolves
    # sensibly from the platform: Apple Silicon -> "mps", explicit cpu -> "cpu".
    def _no_torch(name):
        raise ImportError("no torch in this venv")

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", _no_torch)
    monkeypatch.setattr("scene_worker.training_adapters.sys.platform", "darwin")
    monkeypatch.setattr("scene_worker.training_adapters.platform.machine", lambda: "arm64")

    assert LensLoraTrainer._device_hint(SimpleNamespace(gpu_id="0")) == "mps"
    assert LensLoraTrainer._device_hint(SimpleNamespace(gpu_id="cpu")) == "cpu"


def test_lens_device_hint_falls_back_to_cuda_off_apple_silicon(monkeypatch):
    def _no_torch(name):
        raise ImportError("no torch in this venv")

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", _no_torch)
    monkeypatch.setattr("scene_worker.training_adapters.sys.platform", "linux")
    monkeypatch.setattr("scene_worker.training_adapters.platform.machine", lambda: "x86_64")

    assert LensLoraTrainer._device_hint(SimpleNamespace(gpu_id="1")) == "cuda:1"
    assert LensLoraTrainer._device_hint(SimpleNamespace(gpu_id=None)) == "cuda"


def test_lens_trainer_passes_resolved_device_to_sidecar(tmp_path, monkeypatch):
    # The device the driver resolves must reach the sidecar spec verbatim, so a
    # Mac run actually trains on "mps" instead of erroring on a "cuda" hint.
    import subprocess as _subprocess

    captured = {}

    class _RecordingPopen(_FakeLensSidecarPopen):
        def __init__(self, cmd, env=None, stdout=None, stderr=None):
            spec = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
            captured["device"] = spec["device"]
            super().__init__(cmd, env=env, stdout=stdout, stderr=stderr)

    plan = _lens_train_plan(tmp_path, steps=2)
    monkeypatch.setattr(LensLoraTrainer, "_sidecar_available", lambda self: True)
    monkeypatch.setattr(LensLoraTrainer, "_device_hint", staticmethod(lambda settings: "mps"))
    monkeypatch.setattr(_subprocess, "Popen", _RecordingPopen)

    LensLoraTrainer().train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda *args, **kwargs: None,
        cancel_requested=lambda: False,
    )
    assert captured["device"] == "mps"


def test_resolve_pretrained_source_prefers_loadable_model_dir(tmp_path):
    model_dir = tmp_path / "models" / "z_image"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    source = resolve_pretrained_source(
        {"baseModelPath": str(model_dir), "baseModelRepo": "Tongyi-MAI/Z-Image-Turbo"}
    )

    assert source == str(model_dir)


def test_resolve_pretrained_source_uses_hf_cache_snapshot(tmp_path):
    cache_root = tmp_path / "hub"
    snapshot = write_huggingface_cache_resource(
        cache_root, "Tongyi-MAI/Z-Image-Turbo", "model_index.json", refs_main=True
    )
    repo_root = snapshot.parent.parent

    source = resolve_pretrained_source({"baseModelPath": str(repo_root)})

    assert source == str(snapshot)


def test_resolve_pretrained_source_falls_back_to_repo_when_path_missing(tmp_path):
    source = resolve_pretrained_source(
        {"baseModelPath": str(tmp_path / "absent"), "baseModelRepo": "Tongyi-MAI/Z-Image-Turbo"}
    )

    assert source == "Tongyi-MAI/Z-Image-Turbo"


class FakeTrainingBackend:
    """Stand-in for the torch/diffusers backend so trainer orchestration is
    testable without an inference backend."""

    def __init__(self):
        self.events = []
        self.checkpoints = []
        self.saved = None
        self.cleaned = False

    def loaded_models(self):
        return ["fake/z-image"]

    def load(self, *, settings, plan, config, progress):
        self.events.append("load")

    def prepare_dataset(self, *, items, config, progress, cancel_requested):
        self.events.append("prepare")
        return {"itemCount": len(items), "resolution": bucket_resolution(config.resolution)}

    def train_step(self, *, step, total_steps, config):
        self.events.append(("step", step))
        return 0.5

    def save_checkpoint(self, *, step, output_dir, file_name):
        path = os.path.join(output_dir, f"ckpt-{step}.safetensors")
        self.checkpoints.append(path)
        return path

    def generate_samples(self, *, step, prompts, output_dir, file_name, plan, config):
        self.events.append(("sample", step))
        return [
            {
                "step": step,
                "prompt": prompt,
                "path": os.path.join(output_dir, "samples", f"sample-{index}.png"),
                "relativePath": f"loras/lora_1/samples/sample-{index}.png",
            }
            for index, prompt in enumerate(prompts[:4], start=1)
        ]

    def save_final(self, *, output_dir, file_name):
        self.saved = os.path.join(output_dir, file_name)
        return self.saved

    def cleanup(self):
        self.cleaned = True


def _real_train_plan(tmp_path, *, steps=4, save_every=2, sample_every=0, item_count=1):
    items = []
    for index in range(item_count):
        image = tmp_path / "images" / f"{index:03d}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
        items.append({"imagePath": str(image), "caption": f"miraStyle portrait {index}"})
    return {
        "planVersion": 1,
        "dataset": {"datasetId": "ds_1", "datasetVersion": 3, "items": items},
        "target": {
            "targetId": "z_image_turbo_lora",
            "kernel": "z_image_lora",
            "baseModel": "z_image_turbo",
            "baseModelPath": str(tmp_path / "model"),
        },
        "config": {
            "rank": 16,
            "alpha": 16,
            "learningRate": 0.0001,
            "steps": steps,
            "batchSize": 1,
            "gradientAccumulation": 1,
            "resolution": 1024,
            "saveEvery": save_every,
            "seed": 42,
            "optimizer": "adamw",
            "advanced": {"sampleEvery": sample_every} if sample_every else {},
        },
        "output": {
            "loraId": "lora_1",
            "outputDir": str(tmp_path / "loras" / "lora_1"),
            "fileName": "mira.safetensors",
            "format": "safetensors",
            "triggerWords": ["miraStyle"],
        },
    }


# Mirror crates/sceneworks-core/src/jobs_store.rs::JOB_STATUSES. The Rust API
# rejects any other status with InvalidStatus, which would fail the job mid-run.
_VALID_JOB_STATUSES = {
    "queued",
    "preparing",
    "downloading",
    "loading_model",
    "running",
    "saving",
    "completed",
    "failed",
    "canceled",
    "interrupted",
}


def test_z_image_trainer_runs_stages_checkpoints_and_saves(tmp_path):
    plan = _real_train_plan(tmp_path, steps=4, save_every=2)
    backend = FakeTrainingBackend()
    trainer = ZImageLoraTrainer(backend=backend)
    events_log = []

    result = trainer.train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda status, stage, value, message: events_log.append((status, stage)),
        cancel_requested=lambda: False,
    )

    stages = [stage for _status, stage in events_log]
    statuses = {status for status, _stage in events_log}
    assert backend.events[0] == "load"
    assert ("step", 4) in backend.events
    assert {"loading_model", "caching_latents", "training", "saving"}.issubset(set(stages))
    # Every emitted status must be a valid JobStatus; the Rust API rejects others.
    # In particular, caching runs under "running" (not the invalid "caching").
    assert statuses <= _VALID_JOB_STATUSES
    assert ("running", "caching_latents") in events_log
    assert result["mode"] == "train"
    assert result["stepsCompleted"] == 4
    assert result["outputPath"] == backend.saved == os.path.join(plan["output"]["outputDir"], "mira.safetensors")
    assert result["triggerWords"] == ["miraStyle"]
    # save_every=2, steps=4 -> a single mid-run checkpoint at step 2 (step 4 is final).
    assert backend.checkpoints == [os.path.join(plan["output"]["outputDir"], "ckpt-2.safetensors")]
    assert backend.cleaned is True


def test_z_image_trainer_emits_training_samples_on_sample_cadence(tmp_path):
    plan = _real_train_plan(tmp_path, steps=4, save_every=0, sample_every=2)
    backend = FakeTrainingBackend()
    trainer = ZImageLoraTrainer(backend=backend)
    progress_results = []

    result = trainer.train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda *args: progress_results.append(args[4]) if len(args) > 4 else None,
        cancel_requested=lambda: False,
    )

    assert ("sample", 2) in backend.events
    assert ("sample", 4) in backend.events
    assert len(result["latestTrainingSamples"]) == 4
    assert result["latestTrainingSamples"][0]["step"] == 4
    assert result["samplePrompts"][0].startswith("miraStyle")
    assert result["sampleSettings"] == {
        "numInferenceSteps": 9,
        "guidanceScale": 1.0,
        "sampleSource": "live_adapter",
    }
    sample_updates = [payload for payload in progress_results if payload]
    assert sample_updates[-1]["latestTrainingSamples"][0]["relativePath"].startswith("loras/lora_1/samples/")
    assert sample_updates[-1]["sampleSettings"]["guidanceScale"] == 1.0


def test_z_image_trainer_cancels_and_skips_save(tmp_path):
    plan = _real_train_plan(tmp_path, steps=10, save_every=0)
    backend = FakeTrainingBackend()
    trainer = ZImageLoraTrainer(backend=backend)
    checks = {"count": 0}

    def cancel_requested():
        checks["count"] += 1
        return checks["count"] > 3

    with pytest.raises(InterruptedError):
        trainer.train(
            settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
            plan=plan,
            progress=lambda *args: None,
            cancel_requested=cancel_requested,
        )

    assert backend.saved is None
    assert backend.cleaned is True


def test_ltx_mlx_lora_kernel_is_retired_from_python():
    # Epic 3039 (sc-3049): native MLX LTX LoRA training moved to the Rust mlx-gen
    # engine. The Python kernel was removed, so resolving it now raises — the Rust
    # mlx worker is the sole LTX-training path (routing keeps `ltx_mlx_lora` off
    # non-mlx workers; see jobs_store::training_kernel_is_mlx_only).
    with pytest.raises(TrainingKernelError):
        create_training_kernel("ltx_mlx_lora")


def test_run_lora_train_job_executes_real_run(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None, peaks=None: callback(lambda: False),
    )
    backend = FakeTrainingBackend()
    trainer = ZImageLoraTrainer(backend=backend)
    monkeypatch.setattr("scene_worker.runtime.create_training_kernel", lambda _kernel_id: trainer)

    api = _DryRunApi()
    plan = _real_train_plan(tmp_path, steps=2, save_every=0)
    job = {"id": "job-train-real", "type": "lora_train", "payload": {"dryRun": False, "plan": plan}}

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="0"), job)

    terminal = api.progress[-1]
    assert terminal["status"] == "completed"
    assert terminal["stage"] == "completed"
    assert terminal["result"]["mode"] == "train"
    assert terminal["result"]["fileName"] == "mira.safetensors"
    assert backend.saved is not None


def test_run_lora_train_job_marks_canceled(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None, peaks=None: callback(lambda: False),
    )

    class CancelingTrainer:
        kernel_id = "z_image_lora"

        def loaded_models(self):
            return []

        def train(self, *, settings, plan, progress, cancel_requested):
            raise InterruptedError("LoRA training canceled by user.")

    monkeypatch.setattr("scene_worker.runtime.create_training_kernel", lambda _kernel_id: CancelingTrainer())

    api = _DryRunApi()
    job = {
        "id": "job-train-cancel",
        "type": "lora_train",
        "payload": {"dryRun": False, "plan": {"planVersion": 1, "target": {"kernel": "z_image_lora"}}},
    }

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="0"), job)

    terminal = api.progress[-1]
    assert terminal["status"] == "canceled"
    assert "canceled" in terminal["message"].lower()


def test_run_lora_train_job_reports_friendly_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None, peaks=None: callback(lambda: False),
    )

    class FailingTrainer:
        kernel_id = "z_image_lora"

        def loaded_models(self):
            return []

        def train(self, *, settings, plan, progress, cancel_requested):
            raise RuntimeError("Repository not found: Tongyi-MAI/Z-Image-Turbo")

    monkeypatch.setattr("scene_worker.runtime.create_training_kernel", lambda _kernel_id: FailingTrainer())

    api = _DryRunApi()
    job = {
        "id": "job-train-fail",
        "type": "lora_train",
        "payload": {"dryRun": False, "plan": {"planVersion": 1, "target": {"kernel": "z_image_lora"}}},
    }

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="0"), job)

    terminal = api.progress[-1]
    assert terminal["status"] == "failed"
    assert "model files were not available" in terminal["message"].lower()


def test_main_check_exits_without_api_loop(monkeypatch):
    calls = []
    monkeypatch.setattr("scene_worker.runtime.run_check", lambda settings: calls.append(settings.worker_id))

    main(["--check"])

    assert calls == ["worker-local-0"]


def test_heartbeat_loaded_models_are_not_sent_as_current_job(monkeypatch):
    class Api:
        def __init__(self):
            self.path = None
            self.payload = None

        def post(self, path, payload):
            self.path = path
            self.payload = payload
            return {}

    class Settings:
        worker_id = "worker-1"
        gpu_id = "0"

    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    api = Api()
    heartbeat(api, Settings(), "idle", loaded_models=["model-a"])

    assert api.path == "/api/v1/workers/worker-1/heartbeat"
    assert api.payload == {"status": "idle", "currentJobId": None, "loadedModels": ["model-a"]}


def test_heartbeat_reports_gpu_utilization_when_available(monkeypatch):
    class Api:
        def __init__(self):
            self.payload = None

        def post(self, _path, payload):
            self.payload = payload
            return {}

    class Settings:
        worker_id = "worker-1"
        gpu_id = "0"

    monkeypatch.setattr(
        "scene_worker.runtime.gpu_utilization",
        lambda _gpu_id: {"memoryTotalMb": 24576, "memoryUsedMb": 4096, "memoryFreeMb": 20480, "gpuLoadPercent": 12},
    )

    api = Api()
    heartbeat(api, Settings(), "idle")

    assert api.payload["utilization"] == {
        "memoryTotalMb": 24576,
        "memoryUsedMb": 4096,
        "memoryFreeMb": 20480,
        "gpuLoadPercent": 12,
    }


def test_keepalive_heartbeat_reports_current_loaded_models_each_tick(monkeypatch):
    calls = []
    models_by_tick = [["model-a"], ["model-b"]]
    loaded_model_calls = []

    class StopAfterTwoHeartbeats:
        def __init__(self):
            self.calls = 0

        def wait(self, _interval):
            self.calls += 1
            return self.calls > 2

    class Settings:
        worker_id = "worker-1"
        heartbeat_seconds = 1

    def capture_heartbeat(api, settings, status, current_job_id=None, loaded_models=None):
        calls.append(
            {
                "api": api,
                "settings": settings,
                "status": status,
                "current_job_id": current_job_id,
                "loaded_models": loaded_models,
            }
        )

    monkeypatch.setattr("scene_worker.runtime.heartbeat", capture_heartbeat)

    def loaded_models():
        loaded_model_calls.append("tick")
        return models_by_tick[len(loaded_model_calls) - 1]

    keep_job_alive(
        api=object(),
        settings=Settings(),
        job_id="job-1",
        status="busy",
        stop_event=StopAfterTwoHeartbeats(),
        loaded_models=loaded_models,
    )

    assert [call["status"] for call in calls] == ["busy", "busy"]
    assert [call["current_job_id"] for call in calls] == ["job-1", "job-1"]
    assert [call["loaded_models"] for call in calls] == [["model-a"], ["model-b"]]
    assert len(loaded_model_calls) == 2


def test_keepalive_heartbeat_reports_empty_models_when_source_is_none(monkeypatch):
    calls = []

    class StopAfterOneHeartbeat:
        def wait(self, _interval):
            return bool(calls)

    class Settings:
        worker_id = "worker-1"
        heartbeat_seconds = 1

    def capture_heartbeat(_api, _settings, _status, _current_job_id=None, loaded_models=None):
        calls.append(loaded_models)

    monkeypatch.setattr("scene_worker.runtime.heartbeat", capture_heartbeat)

    keep_job_alive(
        api=object(),
        settings=Settings(),
        job_id="job-1",
        status="busy",
        stop_event=StopAfterOneHeartbeat(),
        loaded_models=None,
    )

    assert calls == [[]]


def test_loaded_model_resolution_failure_keeps_heartbeat_alive(monkeypatch):
    events = []

    def failing_loaded_models():
        raise RuntimeError("cache is mid-load")

    monkeypatch.setattr("scene_worker.runtime.emit", events.append)

    assert resolve_loaded_models(failing_loaded_models, job_id="job-1") == []
    assert events == [
        {
            "event": "loaded_models_failed",
            "error": "cache is mid-load",
            "jobId": "job-1",
            "reportedAt": events[0]["reportedAt"],
        }
    ]


def test_video_job_reports_dynamic_loaded_models_on_progress_and_keepalive(monkeypatch):
    heartbeat_models = []
    blocking_models = []

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                heartbeat_models.append(payload["loadedModels"])
                return {}
            if path.endswith("/progress"):
                return {"status": payload["status"], "stage": payload["stage"]}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def __init__(self):
            self.models = []

        def loaded_models(self):
            return list(self.models)

        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            self.models = ["video-model-loaded"]

        def estimate_requirements(self, _request):
            return {"previewFrames": 1}

        def run(self, *, settings, job, request, progress, cancel_requested):
            self.models = ["video-model-running"]
            progress("running", "generating", 0.5, "Rendering.")
            return {"assetId": "asset-video-1"}

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            raise AssertionError("cleanup should not be called")

    def run_immediately(_api, _settings, _job_id, _status, callback, *, loaded_models, on_force_terminate=None, peaks=None):
        blocking_models.append(loaded_models())
        result = callback(lambda: False)
        blocking_models.append(loaded_models())
        return result

    monkeypatch.setattr("scene_worker.runtime.create_video_adapter", lambda _job=None: VideoAdapter())
    monkeypatch.setattr("scene_worker.runtime.run_blocking_job_step", run_immediately)

    run_video_job(
        Api(),
        SimpleNamespace(worker_id="worker-1"),
        {"id": "job-1", "payload": {"projectId": "project-1", "prompt": "clip"}},
    )

    assert heartbeat_models == [
        [],
        [],
        ["video-model-loaded"],
        ["video-model-running"],
        ["video-model-running"],
    ]
    assert blocking_models == [["video-model-loaded"], ["video-model-running"]]


def test_video_job_estimate_progress_accepts_non_preview_frame_requirements(monkeypatch):
    progress_messages = []

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                return {}
            if path.endswith("/progress"):
                progress_messages.append(payload["message"])
                return {"status": payload["status"], "stage": payload["stage"]}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            return None

        def estimate_requirements(self, _request):
            return {"estimatedFrames": 121, "requestedFrames": 120}

        def run(self, *, settings, job, request, progress, cancel_requested):
            return {"assetId": "asset-video-1"}

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            raise AssertionError("cleanup should not be called")

    monkeypatch.setattr("scene_worker.runtime.create_video_adapter", lambda _job=None: VideoAdapter())
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda *_args, **_kwargs: _args[4](lambda: False),
    )

    run_video_job(
        Api(),
        SimpleNamespace(worker_id="worker-1"),
        {"id": "job-1", "payload": {"projectId": "project-1", "prompt": "clip"}},
    )

    assert "Estimated 121 frames for this clip." in progress_messages


def test_video_job_failure_runs_cleanup_to_free_gpu(monkeypatch):
    events = {"cleanup": 0, "status": None}

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                return {}
            if path.endswith("/progress"):
                events["status"] = payload.get("status")
                return {"status": payload["status"], "stage": payload.get("stage")}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            return None

        def estimate_requirements(self, _request):
            return {"estimatedFrames": 121, "requestedFrames": 120}

        def run(self, *, settings, job, request, progress, cancel_requested):
            raise RuntimeError("CUDA error: out of memory")

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            events["cleanup"] += 1

    monkeypatch.setattr("scene_worker.runtime.create_video_adapter", lambda _job=None: VideoAdapter())
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda *_args, **_kwargs: _args[4](lambda: False),
    )

    # A CUDA OOM cleans up, marks the job failed, then exits (SystemExit) so the
    # supervisor restarts the child with a fresh CUDA context — the poisoned
    # context can't reliably reclaim VRAM in place.
    with pytest.raises(SystemExit):
        run_video_job(
            Api(),
            SimpleNamespace(worker_id="worker-1", gpu_id="0"),
            {"id": "job-oom", "payload": {"projectId": "project-1", "prompt": "clip"}},
        )

    assert events["cleanup"] == 1
    assert events["status"] == "failed"


def test_upscale_and_detail_jobs_restart_after_cuda_oom(monkeypatch):
    # sc-4187: upscale/detail set needs_oom_restart but never restarted — the
    # worker kept running with a poisoned CUDA context. They now release the
    # activation pool and restart like every other GPU handler.
    from scene_worker.runtime import run_detail_job, run_upscale_job

    class Api:
        def __init__(self):
            self.status = None

        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                return {}
            if path.endswith("/progress"):
                self.status = payload.get("status")
                return {"status": payload["status"], "stage": payload.get("stage")}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    for handler, runner_target in (
        (run_upscale_job, "scene_worker.runtime.run_image_upscale"),
        (run_detail_job, "scene_worker.runtime.run_image_detail"),
    ):
        released = {"count": 0}

        def boom(*_args, **_kwargs):
            raise RuntimeError("CUDA error: out of memory")

        monkeypatch.setattr(runner_target, boom)
        monkeypatch.setattr("scene_worker.runtime.find_project_path", lambda *_a, **_k: None)
        monkeypatch.setattr(
            "scene_worker.runtime.release_image_worker_memory",
            lambda: released.__setitem__("count", released["count"] + 1),
        )
        monkeypatch.setattr(
            "scene_worker.runtime.run_blocking_job_step",
            lambda *_args, **_kwargs: _args[4](lambda: False),
        )
        api = Api()
        with pytest.raises(SystemExit):
            handler(
                api,
                SimpleNamespace(worker_id="worker-1", gpu_id="0", data_dir=Path(".")),
                {"id": "job-oom", "payload": {"projectId": "project-1"}},
            )
        assert api.status == "failed"
        assert released["count"] == 1, handler.__name__


def test_video_job_nonoom_failure_does_not_restart(monkeypatch):
    events = {"cleanup": 0, "status": None}

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                return {}
            if path.endswith("/progress"):
                events["status"] = payload.get("status")
                return {"status": payload["status"], "stage": payload.get("stage")}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            return None

        def estimate_requirements(self, _request):
            return {"estimatedFrames": 121, "requestedFrames": 120}

        def run(self, *, settings, job, request, progress, cancel_requested):
            raise RuntimeError("num_frames must be divisible by 8 + 1")

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            events["cleanup"] += 1

    monkeypatch.setattr("scene_worker.runtime.create_video_adapter", lambda _job=None: VideoAdapter())
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda *_args, **_kwargs: _args[4](lambda: False),
    )

    # A non-OOM failure cleans up and marks failed but returns normally (no restart).
    run_video_job(
        Api(),
        SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        {"id": "job-fail", "payload": {"projectId": "project-1", "prompt": "clip"}},
    )

    assert events["cleanup"] == 1
    assert events["status"] == "failed"


def test_random_batch_seeds_are_used_per_image():
    assert resolve_seed(None, "city at night", 2, [101, 202, 303, 404]) == 303


def test_explicit_seed_uses_reproducible_ladder():
    assert resolve_seed(1234, "city at night", 2, [101, 202, 303, 404]) == 1236


def test_video_adapter_override_aliases_and_unknown_values(monkeypatch):
    monkeypatch.delenv("SCENEWORKS_VIDEO_ADAPTER", raising=False)
    # Epic 3018 cutover: the Python worker no longer routes video to MLX (Wan/LTX
    # MLX-eligible jobs are claimed by the Rust GPU worker), so auto-dispatch always
    # lands on the torch adapters by model target.
    assert create_video_adapter({"payload": {"model": "ltx_2_3"}}).__class__.__name__ == "LtxPipelinesVideoAdapter"
    assert create_video_adapter({"payload": {"model": "wan_2_2"}}).__class__.__name__ == "DiffusersVideoAdapter"
    assert create_video_adapter({"payload": {"model": "wan_2_2_t2v_14b"}}).__class__.__name__ == "DiffusersVideoAdapter"
    assert create_video_adapter({"payload": {"model": "wan_2_2_i2v_14b"}}).__class__.__name__ == "DiffusersVideoAdapter"
    assert create_video_adapter().__class__.__name__ == "LtxPipelinesVideoAdapter"

    monkeypatch.setenv("SCENEWORKS_VIDEO_ADAPTER", "procedural")
    assert create_video_adapter().__class__.__name__ == "ProceduralVideoAdapter"

    monkeypatch.setenv("SCENEWORKS_VIDEO_ADAPTER", "ltx_pipelines")
    assert create_video_adapter().__class__.__name__ == "LtxPipelinesVideoAdapter"

    monkeypatch.setenv("SCENEWORKS_VIDEO_ADAPTER", "diffusers_video")
    assert create_video_adapter().__class__.__name__ == "DiffusersVideoAdapter"

    monkeypatch.setenv("SCENEWORKS_VIDEO_ADAPTER", "typo")
    try:
        create_video_adapter()
    except RuntimeError as exc:
        assert "Unsupported SCENEWORKS_VIDEO_ADAPTER" in str(exc)
    else:
        raise AssertionError("Unknown video adapter override should fail loudly.")


def test_create_video_adapter_routes_svd_to_diffusers(monkeypatch):
    # SVD is a diffusers pipeline (not the native LTX stack), so it routes to the
    # generic DiffusersVideoAdapter.
    monkeypatch.delenv("SCENEWORKS_VIDEO_ADAPTER", raising=False)
    adapter = create_video_adapter({"payload": {"model": "svd", "mode": "image_to_video"}})
    assert adapter.__class__.__name__ == "DiffusersVideoAdapter"


def test_svd_video_model_target_defaults():
    target = VIDEO_MODEL_TARGETS["svd"]
    assert target["adapter"] == "svd_video"
    assert target["family"] == "svd"
    # Image-conditioned only — no text_to_video or timeline modes.
    assert target["capabilities"] == ["image_to_video"]
    assert target["repo"] == "stabilityai/stable-video-diffusion-img2vid-xt"
    assert target["variant"] == "fp16"
    # Fixed-length clip defined by the checkpoint.
    assert target["numFrames"] == 25


def test_svd_num_frames_is_fixed_regardless_of_duration():
    adapter = DiffusersVideoAdapter()
    # Duration/fps would imply 60 frames for a 6s clip, but SVD emits its fixed
    # 25-frame burst regardless.
    request = video_request_from_job(
        {"id": "job-svd", "payload": {"projectId": "p", "mode": "image_to_video", "model": "svd", "duration": 6, "fps": 10}}
    )
    assert adapter._num_frames(request) == 25


def test_svd_pipeline_class_resolves_stable_video_diffusion():
    adapter = DiffusersVideoAdapter()
    target = VIDEO_MODEL_TARGETS["svd"]
    request = video_request_from_job({"id": "j", "payload": {"projectId": "p", "mode": "image_to_video", "model": "svd"}})
    fake_diffusers = SimpleNamespace(StableVideoDiffusionPipeline="SVD_PIPE_CLASS")
    assert adapter._pipeline_class(fake_diffusers, request, target) == "SVD_PIPE_CLASS"
    # Missing pipeline class fails loudly rather than silently mis-routing.
    try:
        adapter._pipeline_class(SimpleNamespace(), request, target)
    except RuntimeError as exc:
        assert "StableVideoDiffusionPipeline" in str(exc)
    else:
        raise AssertionError("SVD must require StableVideoDiffusionPipeline.")


def test_svd_pipeline_kwargs_build_image_conditioning_without_prompt(monkeypatch):
    # The SVD branch of _pipeline_kwargs animates the source image with motion
    # controls and passes NO prompt/height/width/guidance (unlike Wan/LTX).
    adapter = DiffusersVideoAdapter()
    target = VIDEO_MODEL_TARGETS["svd"]

    class FakeGen:
        def manual_seed(self, _seed):
            return self

    fake_torch = SimpleNamespace(Generator=lambda _device: FakeGen())
    monkeypatch.setattr(
        "scene_worker.video_adapters.importlib.import_module",
        lambda name: fake_torch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr("scene_worker.video_adapters.select_torch_device", lambda *_a, **_k: "cpu")

    class FakeImage:
        def resize(self, _size):
            return "RESIZED_IMAGE"

    class FakePipe:
        # filter_call_kwargs keeps only named params, so this signature gates
        # which kwargs survive — proving the SVD branch produces these and no prompt.
        def __call__(
            self,
            *,
            image=None,
            num_frames=None,
            num_inference_steps=None,
            decode_chunk_size=None,
            motion_bucket_id=None,
            fps=None,
            noise_aug_strength=None,
            generator=None,
        ):
            return None

    request = video_request_from_job(
        {
            "id": "job-svd",
            "payload": {
                "projectId": "p",
                "mode": "image_to_video",
                "model": "svd",
                "advanced": {"motionBucketId": 90, "conditioningFps": 6},
            },
        }
    )
    kwargs = adapter._pipeline_kwargs(
        pipe=FakePipe(),
        project_path=Path("/tmp/project"),
        request=request,
        target=target,
        first_image=FakeImage(),
        last_image=None,
        seed=7,
        num_frames=25,
    )
    assert kwargs["image"] == "RESIZED_IMAGE"
    assert kwargs["num_frames"] == 25
    assert kwargs["motion_bucket_id"] == 90
    assert kwargs["fps"] == 6
    assert kwargs["noise_aug_strength"] == 0.02
    assert "prompt" not in kwargs
    assert "guidance_scale" not in kwargs


_A14B_QUANT_ENTRY = {
    "quantization": {
        "defaults": {"mps": "gguf-q8_0", "cuda": "gguf-q4_k_m"},
        "variants": {
            "gguf-q8_0": {
                "format": "gguf",
                "label": "GGUF Q8_0",
                "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
                "transformerFile": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf",
                "transformer2File": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf",
            },
            "gguf-q4_k_m": {
                "format": "gguf",
                "label": "GGUF Q4_K_M",
                "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
                "transformerFile": "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
                "transformer2File": "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
            },
        },
    }
}


def _wan_quant_request(advanced=None, manifest=None):
    return video_request_from_job(
        {
            "id": "j",
            "payload": {
                "projectId": "p",
                "mode": "text_to_video",
                "model": "wan_2_2_t2v_14b",
                "advanced": advanced or {},
                "modelManifestEntry": _A14B_QUANT_ENTRY if manifest is None else manifest,
            },
        }
    )


def test_diffusers_wan_gguf_quant_variant_selection():
    adapter = DiffusersVideoAdapter()
    # Explicit selection wins regardless of platform default.
    explicit = adapter._gguf_quant_variant(_wan_quant_request({"quantization": "gguf-q4_k_m"}), "mps")
    assert explicit["id"] == "gguf-q4_k_m"
    assert explicit["transformerFile"].endswith("HighNoise-Q4_K_M.gguf")
    assert explicit["transformer2File"].endswith("LowNoise-Q4_K_M.gguf")
    # Auto / empty falls back to the per-platform default: Q8_0 on MPS, Q4_K_M on CUDA.
    assert adapter._gguf_quant_variant(_wan_quant_request({}), "mps")["id"] == "gguf-q8_0"
    assert adapter._gguf_quant_variant(_wan_quant_request({"quantization": "auto"}), "cuda")["id"] == "gguf-q4_k_m"
    # No default for the platform (cpu) -> unquantized.
    assert adapter._gguf_quant_variant(_wan_quant_request({}), "cpu") is None
    # Explicit opt-out keywords -> unquantized.
    assert adapter._gguf_quant_variant(_wan_quant_request({"quantization": "none"}), "mps") is None
    assert adapter._gguf_quant_variant(_wan_quant_request({"quantization": "full"}), "cuda") is None
    # No quantization block in the manifest entry -> unquantized.
    assert adapter._gguf_quant_variant(_wan_quant_request({}, manifest={}), "mps") is None


def test_diffusers_wan_gguf_injects_high_and_low_experts(monkeypatch):
    adapter = DiffusersVideoAdapter()
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeTransformer:
        @staticmethod
        def from_single_file(path, **kwargs):
            calls.append((path, kwargs))
            return f"T[{path}]"

    fake_diffusers = SimpleNamespace(
        WanTransformer3DModel=FakeTransformer,
        GGUFQuantizationConfig=lambda **kwargs: ("QCFG", kwargs),
    )
    monkeypatch.setattr(adapter, "_resolve_gguf_file", lambda repo, file_name: f"/cache/{repo}/{file_name}")

    # A14B: both experts injected (high -> transformer, low -> transformer_2).
    kwargs: dict[str, Any] = {}
    variant = {
        "id": "gguf-q8_0",
        "format": "gguf",
        "repo": "R",
        "transformerFile": "hi.gguf",
        "transformer2File": "lo.gguf",
    }
    adapter._inject_gguf_experts(fake_diffusers, kwargs, "diffusers/repo", variant, "DT")
    assert kwargs["transformer"] == "T[/cache/R/hi.gguf]"
    assert kwargs["transformer_2"] == "T[/cache/R/lo.gguf]"
    # The diffusers repo is the config source; compute dtype threads to from_single_file.
    assert calls[0][1]["config"] == "diffusers/repo"
    assert calls[0][1]["torch_dtype"] == "DT"

    # 5B dense (single transformer): no transformer_2.
    dense_kwargs: dict[str, Any] = {}
    adapter._inject_gguf_experts(
        fake_diffusers, dense_kwargs, "diffusers/repo", {"format": "gguf", "repo": "R", "transformerFile": "only.gguf"}, "DT"
    )
    assert dense_kwargs["transformer"] == "T[/cache/R/only.gguf]"
    assert "transformer_2" not in dense_kwargs

    # Missing diffusers classes fail loudly rather than silently skipping quantization.
    with pytest.raises(RuntimeError, match="GGUF support"):
        adapter._inject_gguf_experts(SimpleNamespace(), {}, "repo", variant, "DT")


def test_video_pipeline_evicts_previous_pipeline_and_loaded_models():
    adapter = DiffusersVideoAdapter()
    adapter._pipeline = object()
    adapter._pipeline_key_value = "old"
    adapter._loaded_models = {"old-model"}

    class Torch:
        class cuda:
            emptied = False

            @classmethod
            def is_available(cls):
                return True

            @classmethod
            def empty_cache(cls):
                cls.emptied = True

    adapter._evict_pipeline(Torch)

    assert adapter._pipeline is None
    assert adapter._pipeline_key_value is None
    assert adapter.loaded_models() == []
    assert Torch.cuda.emptied is True


def test_ltx_frame_count_uses_nearest_8n_plus_one_value():
    assert ltx_frame_count(100) == 97
    assert ltx_frame_count(150) == 153
    assert ltx_frame_count(200) == 201
    assert ltx_frame_count(250) == 249


def test_ltx_video_requirements_report_normalized_frame_count():
    adapter = DiffusersVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "duration": 6,
                "fps": 25,
                "advanced": {},
            },
        }
    )

    requirements = adapter.estimate_requirements(request)

    assert requirements["requestedFrames"] == 150
    assert requirements["estimatedFrames"] == 153
    assert requirements["repo"] == "Lightricks/LTX-2.3"


def test_ltx_pipelines_multigpu_compat_installs_missing_type_module(monkeypatch):
    for module_name in (
        "ltx_pipelines",
        "ltx_pipelines.multigpu",
        "ltx_pipelines.multigpu.delegating_builder",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    parent = ModuleType("ltx_pipelines")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "ltx_pipelines", parent)

    install_ltx_pipelines_multigpu_compat()

    module = importlib.import_module("ltx_pipelines.multigpu.delegating_builder")
    with pytest.raises(RuntimeError, match="optional multigpu DelegatingBuilder"):
        module.DelegatingBuilder()


def test_require_patch_target_returns_existing_symbol():
    owner = SimpleNamespace(load_weights=lambda self: None)
    target = _require_patch_target(
        owner, "load_weights", pin="some-dep==1.0", patch="example patch"
    )
    assert target is owner.load_weights


def test_require_patch_target_raises_naming_pin_on_missing_symbol():
    owner = SimpleNamespace()
    with pytest.raises(VendorPatchDriftError) as excinfo:
        _require_patch_target(
            owner, "load_weights", pin="mlx-video-with-audio>=0.1.36,<0.2", patch="LTX LoRA wrap (sc-1647)"
        )
    message = str(excinfo.value)
    assert "load_weights" in message
    assert "mlx-video-with-audio>=0.1.36,<0.2" in message
    assert "LTX LoRA wrap (sc-1647)" in message


def test_require_patch_target_raises_when_required_callable_is_not_callable():
    owner = SimpleNamespace(load_weights="not-callable")
    with pytest.raises(VendorPatchDriftError, match="no longer callable"):
        _require_patch_target(
            owner, "load_weights", pin="some-dep==1.0", patch="example patch", require_callable=True
        )


def test_require_patch_target_allows_non_callable_when_not_required():
    owner = SimpleNamespace(some_attr=123)
    assert _require_patch_target(owner, "some_attr", pin="some-dep==1.0", patch="example patch") == 123


def test_video_adapter_tracks_and_discards_temp_outputs(tmp_path):
    # The temp registry lives on the base VideoGenerationAdapter, so the Diffusers
    # adapter (which extends the base directly) gets force-cancel reaping too via the
    # already-wired on_force_terminate hook (sc-1719).
    adapter = DiffusersVideoAdapter()
    first = tmp_path / "a.tmp.mp4"
    second = tmp_path / "b.control.mp4"
    first.write_bytes(b"x")
    second.write_bytes(b"y")
    adapter.track_temp_output("job-1", first)
    adapter.track_temp_output("job-1", second)

    adapter.discard_temp_outputs("job-1")

    assert not first.exists()
    assert not second.exists()
    # Idempotent: the entry is popped, so a later cleanup after the force-cancel hook
    # is a harmless no-op.
    adapter.discard_temp_outputs("job-1")


def test_diffusers_cleanup_reaps_temp_outputs(tmp_path):
    adapter = DiffusersVideoAdapter()
    temp = tmp_path / "clip.tmp.mp4"
    temp.write_bytes(b"x")
    adapter.track_temp_output("job-4", temp)

    adapter.cleanup("job-4")

    assert not temp.exists()


def test_lens_adapter_discards_sidecar_scratch_dir(tmp_path):
    adapter = LensTurboAdapter()
    scratch = tmp_path / "lens_sidecar_abc"
    scratch.mkdir()
    (scratch / "spec.json").write_text("{}", encoding="utf-8")
    adapter._scratch_dir = scratch

    adapter.discard_temp_outputs("job-5")

    assert not scratch.exists()
    assert adapter._scratch_dir is None
    # No scratch dir registered -> no-op.
    adapter.discard_temp_outputs("job-5")


def test_lens_trainer_discards_scratch_dir(tmp_path):
    trainer = LensLoraTrainer()
    scratch = tmp_path / "lens_train_abc"
    scratch.mkdir()
    (scratch / "spec.json").write_text("{}", encoding="utf-8")
    trainer._scratch_dir = scratch

    trainer.discard_temp_outputs()

    assert not scratch.exists()
    # Lazily-set attr cleared; a second call is a harmless no-op.
    trainer.discard_temp_outputs()


class _FakeQuantizationPolicy:
    @staticmethod
    def fp8_cast():
        return "fp8-cast"


def write_native_ltx_manifest(config_dir, *, checkpoint=None, spatial=None, lora=None, gemma=None):
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    resources = {
        "checkpoint": {"repo": "Lightricks/LTX-2.3", "file": "checkpoint.safetensors"},
        "spatialUpscaler": {"repo": "Lightricks/LTX-2.3", "file": "spatial.safetensors"},
        "distilledLora": {"repo": "Lightricks/LTX-2.3", "file": "distilled-lora.safetensors"},
        "gemma": {"repo": "google/gemma-3-12b-it-qat-q4_0-unquantized"},
    }
    if checkpoint is not None:
        resources["checkpoint"] = {"path": str(checkpoint)}
    if spatial is not None:
        resources["spatialUpscaler"] = {"path": str(spatial)}
    if lora is not None:
        resources["distilledLora"] = {"path": str(lora)}
    if gemma is not None:
        resources["gemma"] = {"path": str(gemma)}
    model_entry = {
        "id": "ltx_2_3",
        "name": "LTX-2.3",
        "family": "ltx-video",
        "type": "video",
        "adapter": "ltx_video",
        "capabilities": ["text_to_video", "image_to_video"],
        "downloads": [],
        "paths": {},
        "resources": resources,
        "defaults": {},
        "limits": {},
        "loraCompatibility": {},
        "ui": {},
    }
    (manifest_dir / "builtin.models.jsonc").write_text(
        json.dumps({"schemaVersion": 1, "models": [model_entry]}),
        encoding="utf-8",
    )
    # Rust now resolves+merges the manifest and passes the entry in the job
    # payload as `modelManifestEntry` (story 1653); tests inject this return
    # value so the worker resolves resources without reading the file itself.
    return model_entry


def write_native_ltx_resource_files(tmp_path):
    checkpoint = tmp_path / "checkpoint.safetensors"
    spatial = tmp_path / "spatial.safetensors"
    lora = tmp_path / "distilled-lora.safetensors"
    gemma = tmp_path / "gemma"
    checkpoint.write_bytes(b"checkpoint")
    spatial.write_bytes(b"spatial")
    lora.write_bytes(b"lora")
    gemma.mkdir()
    return checkpoint, spatial, lora, gemma


def write_huggingface_cache_resource(cache_root, repo, file_name=None, revision="abc123", refs_main=False):
    safe_repo = "".join(char if char.isalnum() or char in "._-" else "--" for char in repo).strip("-")
    repo_root = cache_root / f"models--{safe_repo}"
    snapshot = repo_root / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    if refs_main:
        (repo_root / "refs").mkdir(parents=True, exist_ok=True)
        (repo_root / "refs" / "main").write_text(revision, encoding="utf-8")
    if file_name is not None:
        path = snapshot / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file_name.encode("utf-8"))
    return snapshot


def test_native_ltx_adapter_reports_mocked_pipeline_requirements(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "duration": 6,
                "fps": 25,
                "quality": "fast",
                "advanced": {"mockNativeInference": True},
            },
        },
    )

    adapter.ensure_models(request)
    requirements = adapter.estimate_requirements(request)

    assert requirements["adapter"] == "ltx_pipelines"
    assert requirements["pipeline"] == "ltx_pipelines.distilled"
    assert requirements["requestedFrames"] == 150
    assert requirements["estimatedFrames"] == 153
    assert requirements["mockedInference"] is True
    assert requirements["resources"]["checkpointPath"] == str(checkpoint)


def test_native_ltx_pipeline_override_decouples_from_quality(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)

    def pipeline_for(quality, advanced):
        adapter = LtxPipelinesVideoAdapter()
        request = adapter.prepare(
            settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
            job={
                "id": "job-override",
                "payload": {
                    "projectId": "project-1",
                    "mode": "text_to_video",
                    "prompt": "city",
                    "model": "ltx_2_3",
                    "modelManifestEntry": manifest_entry,
                    "duration": 6,
                    "fps": 25,
                    "quality": quality,
                    "advanced": {"mockNativeInference": True, **advanced},
                },
            },
        )
        adapter.ensure_models(request)
        return adapter.estimate_requirements(request)["pipeline"]

    # Distilled override forces single-stage even at balanced quality.
    assert pipeline_for("balanced", {"ltxPipeline": "distilled"}) == "ltx_pipelines.distilled"
    # Two-stage override forces the dev + upscaler path even at fast quality.
    assert pipeline_for("fast", {"ltxPipeline": "two_stage"}) == "ltx_pipelines.ti2vid_two_stages"
    # Auto preserves the quality-driven default.
    assert pipeline_for("balanced", {"ltxPipeline": "auto"}) == "ltx_pipelines.ti2vid_two_stages"
    assert pipeline_for("fast", {}) == "ltx_pipelines.distilled"


def test_native_ltx_precision_selects_quantization_and_offload(monkeypatch):
    import scene_worker.video_adapters as va

    fake_offload = SimpleNamespace(CPU="cpu", DISK="disk", NONE="none")

    def fake_import(name):
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=fake_offload)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.install_ltx_pipelines_multigpu_compat", lambda: None)
    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import)
    adapter = va.LtxPipelinesVideoAdapter()

    def req(advanced):
        return SimpleNamespace(advanced=advanced)

    # Default precision is fp8 -> a quantization policy is built.
    assert adapter._quantization(req({})) == "fp8-cast"
    # Explicit bf16 -> no quantization.
    assert adapter._quantization(req({"precision": "bf16"})) is None
    # Default offload is resident ("none") regardless of precision; CPU streaming
    # leaks/thrashes on this stack, so callers opt into it explicitly.
    assert adapter._offload_mode(req({"precision": "bf16"})) == "none"
    assert adapter._offload_mode(req({})) == "none"
    assert adapter._offload_mode(req({"offloadMode": "cpu"})) == "cpu"
    # The override (used for the torch.compile path) forces resident.
    assert adapter._offload_mode(req({"offloadMode": "cpu"}), override="none") == "none"


def test_native_ltx_distilled_variant_switches_files(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    resources = {
        "checkpoint": {"repo": "Lightricks/LTX-2.3", "file": "ltx-2.3-22b-dev.safetensors"},
        "distilledCheckpoint": {
            "repo": "Lightricks/LTX-2.3",
            "file": "ltx-2.3-22b-distilled-1.1.safetensors",
            "variants": {
                "1.1": "ltx-2.3-22b-distilled-1.1.safetensors",
                "1.0": "ltx-2.3-22b-distilled.safetensors",
            },
        },
        "spatialUpscaler": {"repo": "Lightricks/LTX-2.3", "file": "spatial.safetensors"},
        "distilledLora": {
            "repo": "Lightricks/LTX-2.3",
            "file": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "variants": {
                "1.1": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                "1.0": "ltx-2.3-22b-distilled-lora-384.safetensors",
            },
        },
        "gemma": {"repo": "google/gemma-3-12b-it-qat-q4_0-unquantized"},
    }
    manifest_entry = {
        "id": "ltx_2_3",
        "name": "LTX-2.3",
        "family": "ltx-video",
        "type": "video",
        "adapter": "ltx_video",
        "capabilities": ["text_to_video"],
        "downloads": [],
        "paths": {},
        "resources": resources,
        "defaults": {},
        "limits": {},
        "loraCompatibility": {},
        "ui": {},
    }
    (manifest_dir / "builtin.models.jsonc").write_text(
        json.dumps({"schemaVersion": 1, "models": [manifest_entry]}),
        encoding="utf-8",
    )

    def resolve(quality, advanced):
        adapter = LtxPipelinesVideoAdapter()
        request = adapter.prepare(
            settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
            job={
                "id": "job-variant",
                "payload": {
                    "projectId": "project-1",
                    "mode": "text_to_video",
                    "prompt": "city",
                    "model": "ltx_2_3",
                    "modelManifestEntry": manifest_entry,
                    "duration": 6,
                    "fps": 25,
                    "quality": quality,
                    "advanced": advanced,
                },
            },
        )
        return adapter.resolve_resources(request)

    # Single-stage distilled: the variant selects the checkpoint file.
    assert resolve("fast", {}).checkpoint_path.name == "ltx-2.3-22b-distilled-1.1.safetensors"
    assert resolve("fast", {"distilledVariant": "1.0"}).checkpoint_path.name == "ltx-2.3-22b-distilled.safetensors"
    # Two-stage: the variant selects the distilled LoRA file (dev checkpoint is unversioned).
    two_stage = resolve("balanced", {"distilledVariant": "1.0"})
    assert two_stage.checkpoint_path.name == "ltx-2.3-22b-dev.safetensors"
    assert two_stage.distilled_lora_path.name == "ltx-2.3-22b-distilled-lora-384.safetensors"


def test_native_ltx_missing_resources_reports_all_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-home"))
    manifest_entry = write_native_ltx_manifest(config_dir)
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "advanced": {},
            },
        },
    )

    with pytest.raises(RuntimeError) as exc:
        adapter.ensure_models(request)

    message = str(exc.value)
    assert "checkpointPath" in message
    assert "spatialUpscalerPath" in message
    assert "distilledLoraPath" in message
    assert "gemmaRoot" in message
    assert str(data_dir / "models" / safe_download_dir("Lightricks/LTX-2.3") / "checkpoint.safetensors") in message
    assert str(data_dir / "models" / safe_download_dir("google/gemma-3-12b-it-qat-q4_0-unquantized")) in message


def test_native_ltx_resources_resolve_from_huggingface_cache(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    cache_root = tmp_path / "hf" / "hub"
    data_dir.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))
    manifest_entry = write_native_ltx_manifest(config_dir)
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "checkpoint.safetensors")
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "spatial.safetensors")
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "distilled-lora.safetensors")
    gemma_snapshot = write_huggingface_cache_resource(cache_root, "google/gemma-3-12b-it-qat-q4_0-unquantized", "config.json")
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "advanced": {"mockNativeInference": True},
            },
        },
    )

    adapter.ensure_models(request)
    resources = adapter.estimate_requirements(request)["resources"]

    assert resources["checkpointPath"].endswith("checkpoint.safetensors")
    assert str(cache_root) in resources["checkpointPath"]
    assert resources["spatialUpscalerPath"].endswith("spatial.safetensors")
    assert resources["distilledLoraPath"].endswith("distilled-lora.safetensors")
    assert resources["gemmaRoot"] == str(gemma_snapshot)


def test_native_ltx_resources_resolve_from_mounted_data_cache_without_hf_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    cache_root = data_dir / "cache" / "huggingface" / "hub"
    data_dir.mkdir()
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    manifest_entry = write_native_ltx_manifest(config_dir)
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "checkpoint.safetensors")
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "spatial.safetensors")
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "distilled-lora.safetensors")
    gemma_snapshot = write_huggingface_cache_resource(cache_root, "google/gemma-3-12b-it-qat-q4_0-unquantized", "config.json")
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "advanced": {"mockNativeInference": True},
            },
        },
    )

    adapter.ensure_models(request)
    resources = adapter.estimate_requirements(request)["resources"]

    assert resources["spatialUpscalerPath"].startswith(str(cache_root))
    assert resources["distilledLoraPath"].startswith(str(cache_root))
    assert resources["gemmaRoot"] == str(gemma_snapshot)


def test_native_ltx_fast_pipeline_does_not_require_distilled_lora(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    checkpoint, spatial, _lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, gemma=gemma)
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "quality": "fast",
                "advanced": {"mockNativeInference": True},
            },
        },
    )

    adapter.ensure_models(request)
    requirements = adapter.estimate_requirements(request)

    assert requirements["pipeline"] == "ltx_pipelines.distilled"


def test_native_ltx_advanced_resource_overrides_win(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    write_native_ltx_manifest(config_dir)
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "advanced": {
                    "mockNativeInference": True,
                    "checkpointPath": str(checkpoint),
                    "spatialUpscalerPath": str(spatial),
                    "distilledLoraPath": str(lora),
                    "gemmaRoot": str(gemma),
                },
            },
        },
    )

    adapter.ensure_models(request)
    resources = adapter.estimate_requirements(request)["resources"]

    assert resources["checkpointPath"] == str(checkpoint)
    assert resources["spatialUpscalerPath"] == str(spatial)
    assert resources["distilledLoraPath"] == str(lora)
    assert resources["gemmaRoot"] == str(gemma)


def test_native_ltx_adapter_rejects_unsupported_modes():
    adapter = LtxPipelinesVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "edit_image",
                "prompt": "city",
                "model": "ltx_2_3",
                "advanced": {},
            },
        }
    )

    with pytest.raises(RuntimeError, match="native pipelines currently support"):
        adapter.ensure_models(request)


def test_native_ltx_mocked_run_writes_scene_video_asset(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-ltx",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "Neon harbor",
            "model": "ltx_2_3",
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "advanced": {"mockNativeInference": True},
        },
    }
    monkeypatch.setattr(
        "scene_worker.video_adapters.gradient_frame",
        lambda width, height, _digest: Image.new("RGB", (width, height), "navy"),
    )
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir), job=job)

    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    asset = result["assetWrites"][0]
    media_path = project_path / asset["mediaPath"]
    assert media_path.exists()
    assert result["adapter"] == "ltx_pipelines"
    assert asset["adapter"] == "ltx_pipelines"
    assert asset["rawAdapterSettings"]["pipeline"] == "ltx_pipelines.ti2vid_two_stages"
    assert asset["rawAdapterSettings"]["mockedNativeInference"] is True
    assert adapter.loaded_models() == ["ltx_2_3", "ltx_pipelines.ti2vid_two_stages"]


def test_native_ltx_text_to_video_uses_ltx_pipeline_and_writes_mp4(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    calls = {"init": None, "run": None, "encode": None}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, **kwargs):
            calls["run"] = kwargs
            return ["video-chunk"], "audio-track"

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeGuiderParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_encode_video(**kwargs):
        calls["encode"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={"rename": "map"},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ti2vid_two_stages":
            return SimpleNamespace(TI2VidTwoStagesPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 2,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        if name == "ltx_core.components.guiders":
            return SimpleNamespace(MultiModalGuiderParams=FakeGuiderParams)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    # Pin the CUDA gating recipe so the fp8 default is exercised deterministically.
    # On a host with torch+MPS installed the gating would disable fp8 (it only
    # assumes CUDA when torch is absent, e.g. the CI parity job), so without this
    # the fp8-cast assertion below is host-dependent.
    monkeypatch.setattr(
        adapter,
        "_ltx_device_gating",
        lambda: {
            "device": None,
            "disable_fp8": False,
            "force_offload_none": False,
            "fp32_audio": False,
            "guard_cuda_sync": False,
        },
    )
    job = {
        "id": "job-real-ltx",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "Neon harbor",
            "negativePrompt": "rain",
            "model": "ltx_2_3",
            "modelManifestEntry": manifest_entry,
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "advanced": {"steps": 7, "distilledLoraStrength": 0.6},
        },
    }
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir), job=job)

    adapter.ensure_models(request)
    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    asset = result["assetWrites"][0]
    media_path = project_path / asset["mediaPath"]
    assert media_path.read_bytes() == b"mp4"
    assert calls["init"]["checkpoint_path"] == str(checkpoint)
    assert calls["init"]["distilled_lora"] == [(str(lora), 0.6, {"rename": "map"})]
    # Default precision is fp8 with no torch.compile: quantization is passed and the
    # default offload is resident ("none"); CPU streaming is opt-in.
    assert calls["init"]["quantization"] == "fp8-cast"
    assert calls["init"]["offload_mode"] == "none"
    assert calls["run"]["prompt"] == "Neon harbor"
    assert calls["run"]["negative_prompt"] == "rain"
    assert calls["run"]["num_inference_steps"] == 7
    assert calls["run"]["images"] == []
    assert calls["encode"]["video"] == ["video-chunk"]
    assert calls["encode"]["audio"] == "audio-track"
    assert calls["encode"]["video_chunks_number"] == 2
    assert asset["mimeType"] == "video/mp4"
    assert asset["rawAdapterSettings"]["realModelInference"] is True
    assert asset["rawAdapterSettings"]["mockedNativeInference"] is False
    assert result["requirements"]["mockedInference"] is False


def test_native_ltx_dependency_probe_only_imports_selected_pipeline(monkeypatch):
    imported = []

    def fake_import_module(name):
        imported.append(name)
        if name == "ltx_pipelines.ic_lora":
            raise ImportError(name)
        return SimpleNamespace()

    monkeypatch.setattr("scene_worker.video_adapters.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    monkeypatch.setattr("scene_worker.video_adapters.install_ltx_pipelines_multigpu_compat", lambda: None)

    adapter = LtxPipelinesVideoAdapter()
    text_request = video_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "Neon harbor",
                "model": "ltx_2_3",
                "quality": "balanced",
            }
        }
    )
    ic_request = video_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "image_to_video",
                "prompt": "Neon harbor",
                "model": "ltx_2_3",
                "loras": [{"id": "identity", "icLora": True}],
            }
        }
    )

    assert adapter._dependencies_available(text_request) is True
    assert "ltx_pipelines.ti2vid_two_stages" in imported
    assert "ltx_pipelines.ic_lora" not in imported
    assert adapter._dependencies_available(ic_request) is False


def test_native_ltx_image_to_video_passes_source_image_conditioning(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    image_rel = "assets/images/source.png"
    (project_path / "assets" / "images").mkdir(parents=True)
    Image.new("RGB", (16, 16), "teal").save(project_path / image_rel)
    (project_path / "assets" / "images" / "source.sceneworks.json").write_text(
        json.dumps({"id": "asset-source", "file": {"path": image_rel}}),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    ic_lora = tmp_path / "identity-control.safetensors"
    ic_lora.write_bytes(b"ic-lora")
    calls = {"run": None, "encode": None}

    class FakePipeline:
        def __init__(self, **_kwargs):
            return None

        def __call__(self, **kwargs):
            calls["run"] = kwargs
            return ["video-chunk"], None

    class FakeConditioningInput(NamedTuple):
        path: str
        frame_idx: int
        strength: float

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeGuiderParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_encode_video(**kwargs):
        calls["encode"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ic_lora":
            return SimpleNamespace(ICLoraPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 1,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        if name == "ltx_core.components.guiders":
            return SimpleNamespace(MultiModalGuiderParams=FakeGuiderParams)
        if name == "ltx_pipelines.utils.args":
            return SimpleNamespace(ImageConditioningInput=FakeConditioningInput)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    job = {
        "id": "job-i2v",
        "payload": {
            "projectId": "project-1",
            "mode": "image_to_video",
            "prompt": "Make the harbor move",
            "model": "ltx_2_3",
            "modelManifestEntry": manifest_entry,
            "sourceAssetId": "asset-source",
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "loras": [
                {
                    "id": "identity_ic",
                    "name": "Identity Control",
                    "icLora": True,
                    "installedPath": str(ic_lora),
                    "weight": 0.65,
                    "families": ["ltx-video"],
                }
            ],
            "advanced": {"imageConditioningStrength": 0.7},
        },
    }
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir), job=job)

    adapter.ensure_models(request)
    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    image_condition = calls["run"]["images"][0]
    assert calls["run"]["video_conditioning"] == []
    assert image_condition.path == str(project_path / image_rel)
    assert image_condition.frame_idx == 0
    assert image_condition.strength == 0.7
    assert result["assetWrites"][0]["sourceAssetId"] == "asset-source"
    assert result["assetWrites"][0]["rawAdapterSettings"]["realModelInference"] is True


def test_native_ltx_image_to_video_falls_back_without_ic_lora(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    image_rel = "assets/images/source.png"
    (project_path / "assets" / "images").mkdir(parents=True)
    Image.new("RGB", (16, 16), "teal").save(project_path / image_rel)
    (project_path / "assets" / "images" / "source.sceneworks.json").write_text(
        json.dumps({"id": "asset-source", "file": {"path": image_rel}}),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    style_lora = tmp_path / "cinematic-style.safetensors"
    style_lora.write_bytes(b"style-lora")
    calls = {"init": None, "run": None}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, **kwargs):
            calls["run"] = kwargs
            return ["video-chunk"], None

    class FakeConditioningInput(NamedTuple):
        path: str
        frame_idx: int
        strength: float

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeGuiderParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_encode_video(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={"rename": "map"},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ti2vid_two_stages":
            return SimpleNamespace(TI2VidTwoStagesPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 1,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        if name == "ltx_core.components.guiders":
            return SimpleNamespace(MultiModalGuiderParams=FakeGuiderParams)
        if name == "ltx_pipelines.utils.args":
            return SimpleNamespace(ImageConditioningInput=FakeConditioningInput)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-i2v-missing-lora",
            "payload": {
                "projectId": "project-1",
                "mode": "image_to_video",
                "prompt": "Make the harbor move",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "sourceAssetId": "asset-source",
                "duration": 1,
                "fps": 12,
                "width": 320,
                "height": 256,
                "quality": "balanced",
                "loras": [
                    {
                        "id": "cinematic_style",
                        "name": "Cinematic Style",
                        "installedPath": str(style_lora),
                        "weight": 0.55,
                        "families": ["ltx-video"],
                    }
                ],
                "advanced": {"imageConditioningStrength": 0.75},
            },
        },
    )

    adapter.ensure_models(request)
    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job={"id": "job-i2v-missing-lora"},
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    assert calls["init"]["checkpoint_path"] == str(checkpoint)
    assert calls["init"]["distilled_lora"] == [(str(lora), 0.8, {"rename": "map"})]
    assert calls["init"]["loras"] == ((str(style_lora), 0.55, {"rename": "map"}),)
    assert calls["run"]["images"] == [FakeConditioningInput(str(project_path / image_rel), 0, 0.75)]
    assert "video_conditioning" not in calls["run"]
    assert result["requirements"]["pipeline"] == "ltx_pipelines.ti2vid_two_stages"


def test_native_ltx_extend_clip_uses_ic_lora_video_conditioning(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    video_rel = "assets/videos/source.mp4"
    (project_path / "assets" / "videos").mkdir(parents=True)
    (project_path / video_rel).write_bytes(b"source-video")
    (project_path / "assets" / "videos" / "source.sceneworks.json").write_text(
        json.dumps({"id": "asset-source-video", "type": "video", "file": {"path": video_rel}}),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    ic_lora = tmp_path / "identity-control.safetensors"
    ic_lora.write_bytes(b"ic-lora")
    calls = {"init": None, "run": None, "encode": None}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, **kwargs):
            calls["run"] = kwargs
            return ["video-chunk"], None

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    def fake_encode_video(**kwargs):
        calls["encode"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={"rename": "map"},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ic_lora":
            return SimpleNamespace(ICLoraPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 1,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    job = {
        "id": "job-extend-ic",
        "payload": {
            "projectId": "project-1",
            "mode": "extend_clip",
            "prompt": "Keep the character walking",
            "model": "ltx_2_3",
            "modelManifestEntry": manifest_entry,
            "sourceClipAssetId": "asset-source-video",
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "loras": [
                {
                    "id": "identity_ic",
                    "name": "Identity Control",
                    "icLora": True,
                    "installedPath": str(ic_lora),
                    "weight": 0.7,
                    "families": ["ltx-video"],
                }
            ],
            "advanced": {"videoConditioningStrength": 0.85, "conditioningAttentionStrength": 0.9},
        },
    }
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir), job=job)

    adapter.ensure_models(request)
    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    assert calls["init"]["distilled_checkpoint_path"] == str(checkpoint)
    assert calls["init"]["loras"] == ((str(ic_lora), 0.7, {"rename": "map"}),)
    assert calls["run"]["images"] == []
    assert calls["run"]["video_conditioning"] == [(str(project_path / video_rel), 0.85)]
    assert calls["run"]["conditioning_attention_strength"] == 0.9
    assert result["requirements"]["pipeline"] == "ltx_pipelines.ic_lora"
    assert result["assetWrites"][0]["sourceClipAssetId"] == "asset-source-video"


def test_native_ltx_cleanup_deletes_temp_output_and_evicts_pipeline(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)

    class FakePipeline:
        def __init__(self, **_kwargs):
            return None

        def __call__(self, **_kwargs):
            return ["video-chunk"], None

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeGuiderParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_encode_video(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"partial")
        raise RuntimeError("encoder failed")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ti2vid_two_stages":
            return SimpleNamespace(TI2VidTwoStagesPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 1,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        if name == "ltx_core.components.guiders":
            return SimpleNamespace(MultiModalGuiderParams=FakeGuiderParams)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    job = {
        "id": "job-cleanup",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "Neon harbor",
            "model": "ltx_2_3",
            "modelManifestEntry": manifest_entry,
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "advanced": {},
        },
    }
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir), job=job)

    adapter.ensure_models(request)
    with pytest.raises(RuntimeError, match="encoder failed"):
        adapter.run(
            settings=SimpleNamespace(data_dir=data_dir),
            job=job,
            request=request,
            progress=lambda *_args: None,
            cancel_requested=lambda: False,
        )
    assert list((project_path / "assets" / "videos").rglob("*.tmp.mp4"))

    adapter.cleanup(job["id"])

    assert list((project_path / "assets" / "videos").rglob("*.tmp.mp4")) == []
    assert adapter.loaded_models() == []
    assert adapter._pipeline is None


def test_ltx_video_text_to_video_default_repo_fails_before_diffusers_404():
    adapter = DiffusersVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "advanced": {},
            },
        }
    )

    with pytest.raises(RuntimeError) as exc:
        adapter.ensure_models(request)

    assert "LTX-2.3 text-to-video is supported by the model" in str(exc.value)
    assert "model_index.json" in str(exc.value)


def test_ltx_video_image_modes_keep_image_to_video_diffusers_repo():
    adapter = DiffusersVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "image_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "sourceAssetId": "asset-image",
                "advanced": {},
            },
        }
    )

    requirements = adapter.estimate_requirements(request)

    assert requirements["repo"] == "Lightricks/LTX-Video"


def test_ltx_video_model_repo_override_wins_over_mode_specific_repos():
    adapter = DiffusersVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "advanced": {"modelRepo": "owner/custom-ltx-diffusers"},
            },
        }
    )

    requirements = adapter.estimate_requirements(request)

    assert requirements["repo"] == "owner/custom-ltx-diffusers"
    adapter.ensure_models(request)


def test_evenly_spaced_indices_are_bounded():
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(1, 4) == [0, 0, 0, 0]


def test_frames_from_output_accepts_nested_frames():
    red = Image.new("RGB", (2, 2), "red")
    blue = Image.new("RGB", (2, 2), "blue")

    frames = frames_from_output(SimpleNamespace(frames=[[red, blue]]))

    assert len(frames) == 2
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)


def test_frames_from_output_scales_float_numpy_frames():
    np = pytest.importorskip("numpy")

    # Diffusers video pipelines default to output_type="np": a float32 array in
    # [0, 1] shaped (batch, frames, H, W, 3). PIL rejects float RGB data, so
    # frames_from_output must scale to uint8 instead of raising
    # "Cannot handle this data type: (1, 1, 3), <f4".
    array = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
    array[0, 0, :, :, 0] = 1.0  # first frame fully red

    frames = frames_from_output(SimpleNamespace(frames=array))

    assert len(frames) == 2
    assert frames[0].mode == "RGB"
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)


def test_load_seekable_image_frame_does_not_fallback_on_decompression_bomb(monkeypatch, tmp_path):
    path = tmp_path / "bomb.png"
    path.write_bytes(b"not really used")

    monkeypatch.setattr("scene_worker.video_adapters.Image.open", lambda _path: (_ for _ in ()).throw(Image.DecompressionBombError("too large")))
    monkeypatch.setattr(
        "scene_worker.video_adapters.load_seekable_video_frame",
        lambda _path, _timestamp: (_ for _ in ()).throw(AssertionError("video fallback should not run")),
    )

    assert load_seekable_image_frame(path, 0) is None


def test_person_track_masks_fail_without_track_boxes(tmp_path):
    project_path = tmp_path
    track_dir = project_path / "person-tracks"
    track_dir.mkdir()
    (track_dir / "track_empty.sceneworks.person-track.json").write_text(
        json.dumps({"id": "track_empty", "frames": [], "selectedDetection": {}}),
        encoding="utf-8",
    )

    try:
        person_track_masks(project_path, "track_empty", 64, 64, 2)
    except RuntimeError as exc:
        assert "no usable boxes" in str(exc)
    else:
        raise AssertionError("Empty person tracks should fail loudly.")


def test_character_reference_images_are_capped(tmp_path):
    project_path = tmp_path
    (project_path / "characters").mkdir()
    (project_path / "assets" / "images").mkdir(parents=True)
    references = []
    for index in range(5):
        asset_id = f"asset_ref_{index}"
        media_rel = f"assets/images/ref_{index}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(project_path / media_rel)
        (project_path / f"assets/images/ref_{index}.sceneworks.json").write_text(
            json.dumps({"id": asset_id, "file": {"path": media_rel}}),
            encoding="utf-8",
        )
        references.append({"assetId": asset_id, "approved": True})
    (project_path / "characters" / "character_1.sceneworks.character.json").write_text(
        json.dumps({"id": "character_1", "references": references, "looks": []}),
        encoding="utf-8",
    )

    assert len(character_reference_images(project_path, "character_1", None, 16, 16)) == 4


def test_replace_person_video_result_carries_lineage_facts():
    # sc-1656 slice 3: the worker reports flat video facts and the Rust API builds
    # the sidecar. Pin the facts the worker emits for a replace_person job — the
    # honest personDetection/replacement defaults now live in the Rust builder
    # (project_store::build_generated_asset_sidecar), tested Rust-side.
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "replace_person",
            "prompt": "Replace the hero",
            "model": "wan_2_2",
            "sourceClipAssetId": "asset-video",
            "personTrackId": "track-1",
            "characterId": "character-1",
            "characterLookId": "look-1",
            "replacementMode": "full_person_keep_outfit",
            "advanced": {},
        },
    }
    request = video_request_from_job(job)

    result = video_generation_result(
        request=request,
        target=VIDEO_MODEL_TARGETS["wan_2_2"],
        adapter_id="wan_video",
        asset_id="asset-output",
        generation_set_id="genset-1",
        media_rel="assets/videos/replacement.mp4",
        seed=44,
        created_at="2026-05-17T00:00:00Z",
        mime_type="video/mp4",
        raw_settings={},
    )
    fact = result["assetWrites"][0]
    assert fact["type"] == "video"
    assert fact["mode"] == "replace_person"
    assert fact["mimeType"] == "video/mp4"
    assert fact["personTrackId"] == "track-1"
    assert fact["replacementMode"] == "full_person_keep_outfit"
    assert fact["sourceClipAssetId"] == "asset-video"
    assert fact["characterId"] == "character-1"
    # No real masked-control path ran here, so the worker reports no
    # replacementStatus; the Rust builder fills the honest false defaults.
    assert "replacementStatus" not in fact


def test_format_batch_running_message_names_completed_count_after_first_image():
    assert format_batch_running_message("Z-Image", 0, 4) == "Running Z-Image 1 of 4."
    assert format_batch_running_message("Z-Image", 2, 4) == "Generated 2 of 4. Running Z-Image 3 of 4."


def test_gpu_memory_snapshot_returns_none_when_cuda_unavailable():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    assert gpu_memory_snapshot(Torch, "cuda:0") is None
    assert gpu_memory_snapshot(Torch, "cpu") is None


def test_gpu_memory_snapshot_reports_allocated_and_reserved_bytes():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def memory_allocated(index=None):
                return 50 * 1024 * 1024

            @staticmethod
            def memory_reserved(index=None):
                return 60 * 1024 * 1024

    snapshot = gpu_memory_snapshot(Torch, "cuda:0")

    assert snapshot == {"device": "cuda:0", "allocatedMb": 50.0, "reservedMb": 60.0}


def test_upscaler_engine_selection_is_import_safe_without_torch(monkeypatch):
    imported: list[str] = []

    def fail_torch_import(name):
        imported.append(name)
        if name == "torch":
            raise AssertionError("torch must not be imported while selecting an upscaler")
        return importlib.import_module(name)

    monkeypatch.setattr("scene_worker.upscalers.importlib.import_module", fail_torch_import)

    engine = create_upscaler_engine("real-esrgan")

    assert isinstance(engine, RealESRGANUpscaler)
    assert imported == []


def test_aurasr_engine_selection_is_import_safe_without_torch(monkeypatch):
    imported: list[str] = []

    def fail_torch_import(name):
        imported.append(name)
        if name == "torch":
            raise AssertionError("torch must not be imported while selecting an upscaler")
        return importlib.import_module(name)

    monkeypatch.setattr("scene_worker.upscalers.importlib.import_module", fail_torch_import)

    engine = create_upscaler_engine("aura-sr")

    assert isinstance(engine, AuraSRUpscaler)
    assert imported == []


def test_upscaler_tile_slices_cover_edges_without_overlap_gaps():
    assert tile_slices(5, 3, 2) == [
        TileSlice(0, 0, 2, 2),
        TileSlice(2, 0, 4, 2),
        TileSlice(4, 0, 5, 2),
        TileSlice(0, 2, 2, 3),
        TileSlice(2, 2, 4, 3),
        TileSlice(4, 2, 5, 3),
    ]
    assert tile_slices(5, 3, 0) == [TileSlice(0, 0, 5, 3)]


def test_real_esrgan_upscale_lazily_imports_torch_and_reuses_device_helpers(tmp_path, monkeypatch):
    weights = tmp_path / "realesrgan.pth"
    weights.write_bytes(b"stub")

    class FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"

        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            mps = None

    imports: list[str] = []

    def fake_import_module(name):
        imports.append(name)
        if name == "torch":
            return FakeTorch
        return importlib.import_module(name)

    seen: dict[str, Any] = {}

    def fake_load_model(self, torch, weights_path, *, factor, device, dtype):
        seen.update({"torch": torch, "weights_path": weights_path, "factor": factor, "device": device, "dtype": dtype})
        return object()

    def fake_upscale_with_model(self, torch, model, image, *, factor, device, dtype, tile_size, tile_pad):
        seen.update({"tile_size": tile_size, "tile_pad": tile_pad})
        return image.resize((image.width * factor, image.height * factor))

    monkeypatch.setattr("scene_worker.upscalers.importlib.import_module", fake_import_module)
    monkeypatch.setattr(RealESRGANUpscaler, "_load_model", fake_load_model)
    monkeypatch.setattr(RealESRGANUpscaler, "_upscale_with_model", fake_upscale_with_model)

    result = RealESRGANUpscaler().upscale(
        Image.new("RGB", (3, 4), "white"),
        job=UpscaleJob(factor=2, weights_path=weights, tile_size=64, tile_pad=4),
        settings=SimpleNamespace(gpu_id="cpu"),
    )

    assert result.size == (6, 8)
    assert imports == ["torch"]
    assert seen == {
        "torch": FakeTorch,
        "weights_path": weights,
        "factor": 2,
        "device": "cpu",
        "dtype": "float32",
        "tile_size": 64,
        "tile_pad": 4,
    }


def test_aurasr_upscale_lazily_imports_torch_and_uses_local_weight_file(tmp_path, monkeypatch):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"stub")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            mps = None

    seen: dict[str, Any] = {}

    class FakeAuraModel:
        def upscale_4x_overlapped(self, image, max_batch_size=16, weight_type="checkboard"):
            seen.update({"method": "overlapped", "max_batch_size": max_batch_size, "weight_type": weight_type})
            return image.resize((image.width * 4, image.height * 4))

    class FakeAuraSR:
        @staticmethod
        def from_pretrained(model_id, use_safetensors=True):
            seen.update({"model_id": model_id, "use_safetensors": use_safetensors})
            return FakeAuraModel()

    fake_aura_module = SimpleNamespace(AuraSR=FakeAuraSR)
    imports: list[str] = []

    def fake_import_module(name):
        imports.append(name)
        if name == "torch":
            return FakeTorch
        if name == "aura_sr":
            return fake_aura_module
        return importlib.import_module(name)

    monkeypatch.setattr("scene_worker.upscalers.importlib.import_module", fake_import_module)

    result = AuraSRUpscaler().upscale(
        Image.new("RGB", (3, 4), "white"),
        job=UpscaleJob(factor=4, weights_path=weights, tile_pad=16),
        settings=SimpleNamespace(gpu_id="cpu"),
    )

    assert result.size == (12, 16)
    assert imports == ["torch", "aura_sr"]
    assert seen == {
        "model_id": str(weights),
        "use_safetensors": True,
        "method": "overlapped",
        "max_batch_size": 16,
        "weight_type": "checkboard",
    }


def test_pipeline_component_devices_inspects_known_submodules():
    class Module:
        def __init__(self, device):
            self.device = device

    class Pipe:
        components = {"unet": Module("cuda:0"), "text_encoder": Module("cuda:0"), "vae": Module("cpu")}

    assert pipeline_component_devices(Pipe()) == ["cpu", "cuda:0"]


def test_pipeline_component_devices_falls_back_to_named_attributes():
    class Module:
        def __init__(self, device):
            self.device = device

    class Pipe:
        transformer = Module("cuda:1")
        vae = Module("cuda:1")

    assert pipeline_component_devices(Pipe()) == ["cuda:1"]


def test_verify_pipeline_on_device_raises_when_components_stayed_on_cpu():
    class Module:
        device = "cpu"

    class Pipe:
        components = {"unet": Module(), "vae": Module()}

    with pytest.raises(RuntimeError, match="did not move onto cuda:0"):
        verify_pipeline_on_device(
            Pipe(),
            requested_device="cuda:0",
            model_label="Z-Image-Turbo",
            allow_offload=False,
        )


def test_verify_pipeline_on_device_raises_when_any_component_stayed_on_cpu():
    class Module:
        def __init__(self, device):
            self.device = device

    class Pipe:
        components = {"transformer": Module("cuda:0"), "vae": Module("cpu")}

    with pytest.raises(RuntimeError, match="pipeline components are on cpu, cuda:0"):
        verify_pipeline_on_device(
            Pipe(),
            requested_device="cuda:0",
            model_label="Z-Image-Turbo",
            allow_offload=False,
        )


def test_verify_pipeline_on_device_rejects_wrong_cuda_index():
    class Module:
        device = "cuda:10"

    class Pipe:
        components = {"transformer": Module()}

    with pytest.raises(RuntimeError, match="did not move onto cuda:1"):
        verify_pipeline_on_device(
            Pipe(),
            requested_device="cuda:1",
            model_label="Z-Image-Turbo",
            allow_offload=False,
        )


def test_verify_pipeline_on_device_allows_cpu_offload_layouts():
    class Module:
        device = "cpu"

    class Pipe:
        components = {"unet": Module(), "vae": Module()}

    devices = verify_pipeline_on_device(
        Pipe(),
        requested_device="cuda:0",
        model_label="Z-Image-Turbo",
        allow_offload=True,
    )

    assert devices == ["cpu"]


def test_verify_pipeline_on_device_accepts_matching_cuda_index():
    class Module:
        def __init__(self, device):
            self.device = device

    class Pipe:
        components = {"unet": Module("cuda:0"), "vae": Module("cuda:0")}

    devices = verify_pipeline_on_device(
        Pipe(),
        requested_device="cuda:0",
        model_label="Z-Image-Turbo",
        allow_offload=False,
    )

    assert devices == ["cuda:0"]


def test_emit_worker_event_writes_json_to_stdout(capsys):
    emit_worker_event("image_inference_start", jobId="job-1", imageIndex=2)

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["event"] == "image_inference_start"
    assert payload["jobId"] == "job-1"
    assert payload["imageIndex"] == 2
    assert "reportedAt" in payload


def test_z_image_adapter_emits_phase_diagnostics_and_running_message(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )

    class FakeTransformer:
        device = "cuda:0"

        def parameters(self):
            return iter([])

    class FakePipe:
        components = {"transformer": FakeTransformer(), "vae": FakeTransformer()}

        def __init__(self):
            self.calls = 0

        def to(self, device):
            self._device = device
            return self

        def __call__(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(images=[Image.new("RGB", (8, 8), (10, 20, 30))])

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(repo, **_kwargs):
            return FakePipe()

    class FakeDiffusers:
        ZImagePipeline = FakePipelineClass

        @staticmethod
        def __getattr__(name):
            raise AttributeError(name)

    class FakeTorch:
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def memory_allocated(index=None):
                return 10 * 1024 * 1024

            @staticmethod
            def memory_reserved(index=None):
                return 12 * 1024 * 1024

            @staticmethod
            def set_device(_device):
                return None

            @staticmethod
            def empty_cache():
                return None

        class backends:
            mps = None

        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    real_import = __import__

    def fake_import_module(name):
        if name == "torch":
            return FakeTorch
        if name == "diffusers":
            return FakeDiffusers
        return real_import(name)

    monkeypatch.setattr("scene_worker.image_adapters.importlib.import_module", fake_import_module)

    job = {
        "id": "job-z",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Stormy bridge",
            "model": "z_image_turbo",
            "count": 2,
            "width": 16,
            "height": 16,
        },
    }

    progress_calls: list[dict] = []

    def progress(status, stage, value, message, result=None):
        progress_calls.append(
            {"status": status, "stage": stage, "value": value, "message": message, "result": result}
        )

    adapter = ZImageDiffusersAdapter()
    adapter.generate(
        settings=SimpleNamespace(data_dir=data_dir, gpu_id="0"),
        job=job,
        request=image_request_from_job(job),
        project_path=project_path,
        progress=progress,
        cancel_requested=lambda: False,
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines() if line.strip()]
    event_names = [event["event"] for event in events]
    assert "image_pipeline_load_start" in event_names
    assert "image_pipeline_load_complete" in event_names
    assert "image_pipeline_on_device" in event_names
    assert event_names.count("image_inference_start") == 2
    assert event_names.count("image_inference_complete") == 2
    on_device = next(event for event in events if event["event"] == "image_pipeline_on_device")
    assert on_device["componentDevices"] == ["cuda:0"]
    assert on_device["gpuMemory"]["allocatedMb"] == 10.0

    running_messages = [call["message"] for call in progress_calls if call["status"] == "running"]
    assert running_messages == [
        "Running Z-Image 1 of 2.",
        "Generated 1 of 2. Running Z-Image 2 of 2.",
    ]


def test_z_image_adapter_fails_fast_when_pipeline_stays_on_cpu_with_offload_fallback(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )

    class StuckOnCpu:
        device = "cpu"

    class FakePipe:
        components = {"transformer": StuckOnCpu(), "vae": StuckOnCpu()}

        def to(self, device):
            return self

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(repo, **_kwargs):
            return FakePipe()

    class FakeDiffusers:
        ZImagePipeline = FakePipelineClass

    class FakeTorch:
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def memory_allocated(index=None):
                return 0

            @staticmethod
            def memory_reserved(index=None):
                return 0

            @staticmethod
            def set_device(_device):
                return None

            @staticmethod
            def empty_cache():
                return None

        class backends:
            mps = None

    def fake_import_module(name):
        if name == "torch":
            return FakeTorch
        if name == "diffusers":
            return FakeDiffusers
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.image_adapters.importlib.import_module", fake_import_module)

    job = {
        "id": "job-cpu-stuck",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Misty harbor",
            "model": "z_image_turbo",
            "count": 1,
            "width": 16,
            "height": 16,
            "advanced": {"cpuOffload": True},
        },
    }

    adapter = ZImageDiffusersAdapter()

    with pytest.raises(RuntimeError, match="did not move onto"):
        adapter.generate(
            settings=SimpleNamespace(data_dir=data_dir, gpu_id="0"),
            job=job,
            request=image_request_from_job(job),
            project_path=project_path,
            progress=lambda *_args, **_kwargs: None,
            cancel_requested=lambda: False,
        )


def test_edit_run_pipeline_threads_project_path(tmp_path, monkeypatch):
    """Regression: ZImage/Qwen ._run_pipeline must accept project_path.

    sc-1678's resolver-threading refactor left project_path referenced inside the
    edit_image branch (load_source_image(project_path, request)) without making it
    a parameter — a latent NameError that only fires at real img2img inference,
    which the mocked suite never exercises. KolorsDiffusersAdapter is the reference
    pattern. This guards both the signature and the live edit-branch call site.
    """
    import inspect

    # Signature guard: project_path must precede the optional cancel_requested,
    # because every call site passes it positionally.
    for adapter_cls in (ZImageDiffusersAdapter, QwenImageAdapter, SdxlDiffusersAdapter):
        params = list(inspect.signature(adapter_cls._run_pipeline).parameters)
        assert "project_path" in params, f"{adapter_cls.__name__}._run_pipeline must accept project_path"
        assert params.index("project_path") < params.index("cancel_requested")

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen_paths: list[Path] = []

    def fake_load_source_image(project_path, _request):
        seen_paths.append(project_path)
        return FakeImage()

    monkeypatch.setattr("scene_worker.image_adapters.load_source_image", fake_load_source_image)

    project_path = tmp_path / "project"
    project_path.mkdir()

    # Runtime guard: invoke the edit_image branch directly (gpu_id="cpu" keeps
    # device selection off real torch) so project_path resolves at the exact line
    # that raised NameError instead of crashing.
    for adapter, model in (
        (ZImageDiffusersAdapter(), "z_image_edit"),
        (QwenImageAdapter(), "qwen_image_edit"),
        (SdxlDiffusersAdapter(), "sdxl"),
    ):
        seen_paths.clear()
        request = image_request_from_job(
            {
                "payload": {
                    "projectId": "project-1",
                    "mode": "edit_image",
                    "model": model,
                    "prompt": "repaint the sky",
                    "sourceAssetId": "asset-source",
                    "width": 16,
                    "height": 16,
                    "count": 1,
                }
            }
        )
        result = adapter._run_pipeline(
            SimpleNamespace(gpu_id="cpu"),
            FakePipe(),
            request,
            7,
            project_path,
        )
        assert seen_paths == [project_path], f"{adapter.id} edit branch must call load_source_image(project_path, ...)"
        assert result is FakeOutput.images[0]


def test_diffusers_image_adapters_call_apply_sampler(tmp_path, monkeypatch):
    """Epic 1753 sc-1762: Z-Image and Qwen ``_run_pipeline`` must thread the
    sampler/scheduler/shift selection into ``apply_sampler`` so the user's
    choice actually swaps the pipeline scheduler. A regression here would
    silently ignore the advanced fields and leave the model on its default
    scheduler — exactly the bug class this epic exists to prevent.
    """

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        scheduler = None

        def __call__(self, **kwargs):
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    captured: list[tuple[str, ...]] = []

    def fake_apply_sampler(pipe, sampler, scheduler, shift, *, adapter=None):
        captured.append((adapter, sampler, scheduler, shift))
        return {"sampler": sampler, "scheduler": scheduler, "shift": shift, "noop": False}

    monkeypatch.setattr("scene_worker.image_adapters.apply_sampler", fake_apply_sampler)

    for adapter, model in (
        (ZImageDiffusersAdapter(), "z_image_turbo"),
        (QwenImageAdapter(), "qwen_image"),
    ):
        captured.clear()
        request = image_request_from_job(
            {
                "payload": {
                    "projectId": "project-1",
                    "mode": "text_to_image",
                    "model": model,
                    "prompt": "still life with sampler swap",
                    "width": 16,
                    "height": 16,
                    "count": 1,
                    "advanced": {
                        "sampler": "dpmpp",
                        "scheduler": "karras",
                    },
                }
            }
        )
        adapter._run_pipeline(
            SimpleNamespace(gpu_id="cpu"),
            FakePipe(),
            request,
            7,
            tmp_path,
        )
        assert captured == [(adapter.id, "dpmpp", "karras", None)], (
            f"{adapter.id} did not thread sampler/scheduler into apply_sampler"
        )

    # Shift axis: scheduler == "shift" must pass through a numeric shift value.
    captured.clear()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": "shift test",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {
                    "sampler": "euler",
                    "scheduler": "shift",
                    "schedulerShift": 4.5,
                },
            }
        }
    )
    ZImageDiffusersAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"),
        FakePipe(),
        request,
        9,
        tmp_path,
    )
    assert captured == [("z_image_diffusers", "euler", "shift", pytest.approx(4.5))]


# --- Cancellation: JobCancelMonitor watchdog + per-step interrupt ---------------


class _CancelPollApi:
    """Fake API for JobCancelMonitor tests: GET reports a (mutable) cancel flag;
    POST /progress records the payload so the test can assert the final status."""

    def __init__(self, cancel_requested=False):
        self.cancel_requested = cancel_requested
        self.progress = []

    def get(self, _path):
        return {"cancelRequested": self.cancel_requested}

    def post(self, path, payload):
        if path.endswith("/heartbeat"):
            return {}
        if path.endswith("/progress"):
            self.progress.append(payload)
            return {"status": payload["status"], "stage": payload["stage"]}
        raise AssertionError(path)


def _wait_until(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_cancel_monitor_caches_cancel_state(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    api = _CancelPollApi(cancel_requested=True)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=0)
    monitor = JobCancelMonitor(api, settings, "job-1", poll_interval=0.05, force_cancel_seconds=0)
    monitor.start()
    try:
        assert _wait_until(monitor.requested)
    finally:
        monitor.stop()


def test_cancel_monitor_does_not_escalate_without_cancel(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    exits = []
    monkeypatch.setattr("scene_worker.runtime.os._exit", lambda code: exits.append(code))
    api = _CancelPollApi(cancel_requested=False)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=0.1)
    monitor = JobCancelMonitor(api, settings, "job-1", poll_interval=0.02, force_cancel_seconds=0.1)
    monitor.start()
    try:
        time.sleep(0.3)
        assert exits == []
        assert monitor.requested() is False
    finally:
        monitor.stop()


def test_cancel_monitor_force_terminates_after_deadline(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    exited = threading.Event()
    exits = []

    def fake_exit(code):
        exits.append(code)
        exited.set()

    monkeypatch.setattr("scene_worker.runtime.os._exit", fake_exit)
    api = _CancelPollApi(cancel_requested=True)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=0.15)
    monitor = JobCancelMonitor(api, settings, "job-9", poll_interval=0.02, force_cancel_seconds=0.15)
    monitor.start()
    try:
        assert exited.wait(timeout=2.0)
    finally:
        monitor.stop()
    assert exits == [FORCED_CANCEL_EXIT_CODE]
    # The job was marked canceled before the (mocked) process exit, so the UI
    # resolves immediately instead of waiting for a stale-worker reaper.
    terminal = api.progress[-1]
    assert terminal["status"] == "canceled"
    assert terminal["stage"] == "canceled"
    assert "force-stopped" in terminal["message"].lower()


def test_cancel_monitor_runs_on_force_terminate_hook_before_exit(monkeypatch):
    # The force-cancel backstop reaps tracked temp files via on_force_terminate
    # before the hard os._exit (which skips cooperative adapter cleanup) — the
    # image/training runners rely on this to drop their scratch dirs (sc-1719).
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    events: list[tuple[str, object]] = []
    exited = threading.Event()

    def fake_exit(code):
        events.append(("exit", code))
        exited.set()

    monkeypatch.setattr("scene_worker.runtime.os._exit", fake_exit)
    api = _CancelPollApi(cancel_requested=True)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=0.1)
    monitor = JobCancelMonitor(
        api,
        settings,
        "job-hook",
        poll_interval=0.02,
        force_cancel_seconds=0.1,
        on_force_terminate=lambda: events.append(("hook", None)),
    )
    monitor.start()
    try:
        assert exited.wait(timeout=2.0)
    finally:
        monitor.stop()
    assert ("hook", None) in events
    # The reap must run before the process exits.
    assert events.index(("hook", None)) < events.index(("exit", FORCED_CANCEL_EXIT_CODE))


def test_cancel_monitor_skips_escalation_when_stopped(monkeypatch):
    # A cooperative cancel that finishes before the deadline must not force-kill.
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    exits = []
    monkeypatch.setattr("scene_worker.runtime.os._exit", lambda code: exits.append(code))
    api = _CancelPollApi(cancel_requested=True)
    settings = SimpleNamespace(worker_id="worker-1", force_cancel_seconds=5)
    monitor = JobCancelMonitor(api, settings, "job-2", poll_interval=0.02, force_cancel_seconds=5)
    monitor.start()
    assert _wait_until(monitor.requested)  # cancel observed
    monitor.stop()  # cooperative path finished the job before the deadline
    time.sleep(0.1)
    assert exits == []


class _FakeStepPipe:
    def __init__(self):
        self._interrupt = False

    def __call__(self, *, prompt=None, callback_on_step_end=None):
        return None


class _NoCallbackPipe:
    def __call__(self, *, prompt=None):
        return None


def test_cancel_step_callback_sets_interrupt_when_canceled():
    pipe = _FakeStepPipe()
    callback = cancel_step_callback(pipe, lambda: True)
    assert callback is not None
    returned = callback(pipe, 0, None, {"latents": "x"})
    assert returned == {"latents": "x"}  # callback must pass kwargs through
    assert pipe._interrupt is True


def test_cancel_step_callback_leaves_interrupt_when_not_canceled():
    pipe = _FakeStepPipe()
    callback = cancel_step_callback(pipe, lambda: False)
    assert callback is not None
    callback(pipe, 0, None, {})
    assert pipe._interrupt is False


def test_cancel_step_callback_none_without_predicate():
    assert cancel_step_callback(_FakeStepPipe(), None) is None


def test_cancel_step_callback_none_when_pipe_lacks_support():
    assert cancel_step_callback(_NoCallbackPipe(), lambda: True) is None


def test_worker_settings_force_cancel_seconds_default(monkeypatch):
    monkeypatch.delenv("SCENEWORKS_WORKER_FORCE_CANCEL_SECONDS", raising=False)
    from scene_worker.settings import WorkerSettings

    assert WorkerSettings().force_cancel_seconds == 30


def test_worker_settings_force_cancel_seconds_env_override(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_WORKER_FORCE_CANCEL_SECONDS", "5")
    from scene_worker.settings import WorkerSettings

    assert WorkerSettings().force_cancel_seconds == 5


def test_repo_slug_functions_match_cross_language_contract():
    # sc-1667: the repo->directory slug string ops are duplicated in the Rust
    # API, the Rust CPU worker, and here. They must stay byte-identical so a
    # repo resolves to the same managed dir / HF cache dir in every language.
    # This fixture is the shared contract; matching Rust tests live in
    # apps/rust-api and crates/sceneworks-worker.
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "rust_migration_contracts" / "repo_slugs.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
    assert cases, "repo_slugs fixture has no cases"
    for case in cases:
        repo = case["repo"]
        assert safe_download_dir(repo) == case["safeDownloadDir"], f"safe_download_dir drift for {repo!r}"
        assert safe_repo_dir_name(repo) == case["safeRepoDirName"], f"safe_repo_dir_name drift for {repo!r}"


# ---- Prompt refinement (sc-2041) ----


def _refine_settings(**overrides):
    base = {
        "worker_id": "worker-1",
        "gpu_id": "cpu",
        "prompt_refine_model": "",
        "prompt_refine_max_new_tokens": 512,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeRefiner:
    """Stands in for PromptRefiner in handler tests — no torch/transformers."""

    id = "prompt_refiner"
    instances = []

    def __init__(self, *, model_name_or_path, gpu_id, max_new_tokens):
        self.model_name_or_path = model_name_or_path
        self.gpu_id = gpu_id
        self.max_new_tokens = max_new_tokens
        self.loaded = False
        self.load_calls = 0
        _FakeRefiner.instances.append(self)

    def loaded_models(self):
        return [self.model_name_or_path] if self.loaded and self.model_name_or_path else []

    def load(self):
        self.load_calls += 1
        self.loaded = True

    def unload(self):
        freed = self.loaded
        self.loaded = False
        return freed

    def refine(self, prompt, *, guide, workflow):
        return f"Refined ({workflow}): {prompt}"


def _refine_adapters(**overrides):
    """A worker-loop-style adapter dict holding a single resident _FakeRefiner."""
    refiner = _FakeRefiner(model_name_or_path=overrides.get("model_name_or_path", ""), gpu_id="cpu", max_new_tokens=512)
    return {"prompt_refiner": refiner}


def test_build_system_prompt_uses_workflow_medium_and_embeds_guide():
    image = build_system_prompt("# Z-Image Guide\n\nUse short prompts.", "image")
    assert "generative image model" in image
    assert "Z-Image Guide" in image

    video = build_system_prompt(None, "video")
    assert "generative video model" in video
    assert "# Model prompt guide" not in video


def test_clean_output_strips_reasoning_and_quoting():
    assert clean_output("<think>plan</think>A vivid sunset over hills.") == "A vivid sunset over hills."
    assert clean_output('"A vivid sunset over hills."') == "A vivid sunset over hills."
    assert clean_output("```\nA vivid sunset over hills.\n```") == "A vivid sunset over hills."


def test_prompt_refiner_load_unavailable_without_backend():
    # torch/transformers aren't installed in the CI worker-test env, so load()
    # surfaces a clear PromptRefineUnavailable rather than an opaque ImportError.
    with pytest.raises(PromptRefineUnavailable):
        PromptRefiner(model_name_or_path="some/model", gpu_id="cpu").load()


def test_prompt_refiner_load_is_idempotent_when_already_resident(monkeypatch):
    # sc-4191: a second load() for the same checkpoint must not re-import torch /
    # re-run from_pretrained — it returns immediately while the model is resident.
    refiner = PromptRefiner(model_name_or_path="some/model", gpu_id="cpu")
    refiner.model = object()  # pretend the checkpoint is resident
    refiner._loaded_model_name = "some/model"

    def _boom(*_args, **_kwargs):
        raise AssertionError("load() must not reload while the same model is resident")

    monkeypatch.setattr("scene_worker.prompt_refine.require_inference_backend_for_gpu_worker", _boom)
    refiner.load()  # no exception → early-returned without touching the backend


def test_prompt_refiner_unload_frees_and_is_safe_when_empty():
    # sc-4191: unload() drops the resident model and reports it; a no-op when
    # nothing is loaded so evict_other_image_adapters can call it unconditionally.
    refiner = PromptRefiner(model_name_or_path="some/model", gpu_id="cpu")
    assert refiner.unload() is False  # nothing loaded yet

    released = {}

    refiner.model = object()
    refiner.tokenizer = object()
    refiner.torch = object()
    refiner._loaded_model_name = "some/model"
    import scene_worker.prompt_refine as pr

    orig = pr.release_inference_memory
    pr.release_inference_memory = lambda torch: released.setdefault("called", True)
    try:
        assert refiner.unload() is True
    finally:
        pr.release_inference_memory = orig

    assert released.get("called") is True
    assert refiner.model is None
    assert refiner.tokenizer is None
    assert refiner.loaded_models() == []


def test_prompt_refiner_refine_applies_chat_template_and_cleans():
    # Drive refine() with a fake tokenizer/model so we exercise the chat-template
    # → generate → decode → clean path without real weights.
    refiner = PromptRefiner(model_name_or_path="some/model", gpu_id="cpu")

    class _Ids:
        shape = (1, 3)

        def to(self, _device):
            return self

    captured = {}

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            return _Ids()

        def decode(self, tokens, **kwargs):
            return "<think>scheming</think>A vivid neon street at midnight."

    class _Model:
        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return [[101, 102, 103, 201, 202]]

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    refiner.tokenizer = _Tokenizer()
    refiner.model = _Model()
    refiner.device = "cpu"
    refiner.torch = SimpleNamespace(no_grad=lambda: _NoGrad())

    out = refiner.refine("neon street", guide="# Guide", workflow="image")

    assert out == "A vivid neon street at midnight."  # think-block stripped
    # Instruction + guide folded into a single user turn (portable across templates).
    assert captured["messages"][0]["role"] == "user"
    assert "# Guide" in captured["messages"][0]["content"]
    assert "neon street" in captured["messages"][0]["content"]


def test_run_prompt_refine_job_writes_refined_result(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    _FakeRefiner.instances = []
    adapters = _refine_adapters()
    api = _DryRunApi()
    job = {"id": "job-refine-1", "type": "prompt_refine", "payload": {"prompt": "dog in park", "workflow": "image"}}

    run_prompt_refine_job(api, _refine_settings(), job, adapters)

    terminal = api.progress[-1]
    assert terminal["status"] == "completed"
    assert terminal["result"]["refinedPrompt"] == "Refined (image): dog in park"
    assert terminal["result"]["originalPrompt"] == "dog in park"
    # Loaded the model before refining; emitted a loading_model stage.
    assert adapters["prompt_refiner"].loaded is True
    assert any(entry["stage"] == "loading_model" for entry in api.progress)


def test_run_prompt_refine_job_reuses_resident_refiner(monkeypatch):
    """sc-4191: repeat prompt_refine jobs must reuse the one resident refiner and
    its already-loaded model, never construct a new instance or reload per job."""
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    _FakeRefiner.instances = []
    adapters = _refine_adapters()
    refiner = adapters["prompt_refiner"]
    api = _DryRunApi()

    for n in range(3):
        job = {"id": f"job-refine-reuse-{n}", "type": "prompt_refine", "payload": {"prompt": "dog", "workflow": "image"}}
        run_prompt_refine_job(api, _refine_settings(), job, adapters)

    # Exactly one refiner ever constructed across three jobs.
    assert len(_FakeRefiner.instances) == 1
    assert _FakeRefiner.instances[0] is refiner
    # load() is called each job but stays cheap (idempotent in the real adapter);
    # the fake just records it. The instance is reused, not rebuilt.
    assert refiner.load_calls == 3
    assert refiner.loaded is True


def test_run_prompt_refine_job_reports_failure(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)

    class _BoomRefiner(_FakeRefiner):
        def refine(self, prompt, *, guide, workflow):
            raise PromptRefineUnavailable("Could not load the prompt-refinement model.")

    api = _DryRunApi()
    adapters = {"prompt_refiner": _BoomRefiner(model_name_or_path="", gpu_id="cpu", max_new_tokens=512)}
    job = {"id": "job-refine-2", "type": "prompt_refine", "payload": {"prompt": "dog", "workflow": "image"}}

    run_prompt_refine_job(api, _refine_settings(), job, adapters)

    terminal = api.progress[-1]
    assert terminal["status"] == "failed"
    assert "Could not load" in terminal["message"]


# ---------------------------------------------------------------------------
# Multi-model Character Studio reference matrix (epic 2003 / sc-2018)
# ---------------------------------------------------------------------------

import re as _matrix_re


def _strip_jsonc_comments(body: str) -> str:
    """Mirror scripts/check-scaffold.mjs::stripJsoncComments so the audit reads
    the real `config/manifests/builtin.models.jsonc` without a JSONC dependency.
    Walks the body char-by-char, suppressing // line and /* block */ comments
    but leaving them intact when they appear inside string literals.
    """
    result: list[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(body):
        char = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue
        if char == "/" and nxt == "/":
            while i < len(body) and body[i] != "\n":
                i += 1
            result.append("\n")
            continue
        if char == "/" and nxt == "*":
            i += 2
            while i < len(body) - 1 and not (body[i] == "*" and body[i + 1] == "/"):
                i += 1
            i += 2
            continue
        result.append(char)
        i += 1
    return "".join(result)


def _load_builtin_models_manifest() -> dict:
    manifest_path = Path(__file__).resolve().parent.parent / "config" / "manifests" / "builtin.models.jsonc"
    raw = manifest_path.read_text(encoding="utf-8")
    return json.loads(_strip_jsonc_comments(raw))


def test_character_image_capability_implies_engine_or_tuning_declaration():
    """Every builtin model that advertises `character_image` must have either
    a worker engine block (`ipAdapter` / `instantId` in MODEL_TARGETS) OR a
    `ui.variationStrength` declaration in the manifest. Otherwise the capability
    flag is dishonest — the picker shows the model in "With character" mode but
    the worker silently ignores the reference, the same shape as z_image_turbo's
    pre-sc-2005 bug. This is the cross-backbone guard for epic 2003 (sc-2018):
    adding a future character_image backbone without engine wiring will fail
    here before it ever reaches a user.
    """
    manifest = _load_builtin_models_manifest()
    misleading: list[str] = []
    for model in manifest.get("models", []):
        capabilities = model.get("capabilities") or []
        if "character_image" not in capabilities:
            continue
        target = MODEL_TARGETS.get(model["id"], {})
        ui = model.get("ui") or {}
        has_engine = bool(target.get("ipAdapter") or target.get("instantId") or target.get("pulidFlux"))
        has_variation_ui = bool(ui.get("variationStrength"))
        if not (has_engine or has_variation_ui):
            misleading.append(model["id"])
    assert not misleading, (
        f"Models advertise `character_image` without engine wiring or a "
        f"`ui.variationStrength` declaration: {misleading}. Add an `ipAdapter`, "
        f"`instantId`, or `pulidFlux` block in MODEL_TARGETS for an IP-Adapter / "
        f"face-ID backbone, or declare `ui.variationStrength` for an edit-style "
        f"backbone (sc-2017), or drop the capability flag (the z_image_turbo bug, "
        f"sc-2005)."
    )


def test_kolors_declares_strict_pose_controlnet():
    """sc-2264: Kolors is the strict pose tier — the manifest must advertise
    ui.poseLibrary AND the worker target must carry the controlNetPose config so
    the pose picker offers it and the adapter can load the pose ControlNet."""
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    kolors = manifest_by_id.get("kolors", {})
    assert kolors.get("ui", {}).get("poseLibrary") is True, (
        "kolors must declare ui.poseLibrary so the pose picker offers the strict tier (sc-2264)."
    )
    target = MODEL_TARGETS.get("kolors", {})
    assert target.get("controlNetPose", {}).get("repo") == "Kwai-Kolors/Kolors-ControlNet-Pose", (
        "kolors MODEL_TARGETS must carry the Kolors-ControlNet-Pose repo for the strict pose path."
    )
    # Identity still rides the IP-Adapter; the pose path composes both.
    assert target.get("ipAdapter"), "kolors pose path needs the IP-Adapter for identity."


def test_models_with_engine_block_advertise_character_image():
    """The reverse-drift guard. Any model that ships an `ipAdapter` or
    `instantId` block in MODEL_TARGETS exists to serve Character Studio's
    reference flow — the manifest must advertise the capability so the picker
    surfaces it. Catches the case where someone wires the worker engine but
    forgets to flip the manifest flag, leaving the engine unreachable.
    """
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    unreachable: list[str] = []
    for model_id, target in MODEL_TARGETS.items():
        if not (target.get("ipAdapter") or target.get("instantId") or target.get("pulidFlux")):
            continue
        builtin = manifest_by_id.get(model_id)
        if builtin is None:
            # Worker-only target not exposed as a built-in (unwired path).
            continue
        capabilities = builtin.get("capabilities") or []
        if "character_image" not in capabilities:
            unreachable.append(model_id)
    assert not unreachable, (
        f"Models have engine blocks in MODEL_TARGETS but the builtin manifest "
        f"does not advertise `character_image`: {unreachable}. Add the capability "
        f"to `capabilities` and `ui.recommendedFor` so the Image Studio "
        f"\"With character\" picker surfaces the model."
    )


def test_hide_reference_strength_models_declare_a_variation_knob():
    """Symmetry guard for the sc-2017 picker UX. A model that opts out of the
    IP-Adapter reference-strength slider via `ui.hideReferenceStrength` MUST
    also declare `ui.variationStrength` — otherwise the picker shows no tuning
    control at all, and the worker silently runs at default true_cfg_scale.
    """
    manifest = _load_builtin_models_manifest()
    unbalanced: list[str] = []
    for model in manifest.get("models", []):
        ui = model.get("ui") or {}
        if not ui.get("hideReferenceStrength"):
            continue
        if not ui.get("variationStrength"):
            unbalanced.append(model["id"])
    assert not unbalanced, (
        f"Models hide the Reference-strength slider without declaring "
        f"`ui.variationStrength`: {unbalanced}. The picker would leave the user "
        f"with NO identity tuning control. Add `variationStrength` or drop "
        f"`hideReferenceStrength`."
    )


# ---- sc-2003: multi-backbone angle-set picker ----


def test_character_studio_angle_prompt_augments_cover_full_pack():
    """Every angle in the canonical ANGLE_SET_ORDER must have a matching
    prompt-augment string. Backbones that consume the shared list (Qwen, FLUX2,
    SenseNova) won't render correctly for an angle without an augment — the
    user would get an unaugmented prompt repeated for that index."""
    from scene_worker.character_studio_angles import (
        ANGLE_PROMPT_AUGMENTS,
        ANGLE_SET_ORDER,
        augment_prompt_for_angle,
    )

    missing = [angle for angle in ANGLE_SET_ORDER if not ANGLE_PROMPT_AUGMENTS.get(angle)]
    assert not missing, f"Angles missing prompt augments: {missing}"
    # Augment helper appends the per-angle clause to the user's base prompt.
    augmented = augment_prompt_for_angle("a portrait of the woman", "three_quarter_left")
    assert "three-quarter left profile" in augmented
    assert augmented.startswith("a portrait of the woman,")
    # Unknown angle is a no-op (caller passes the base through unchanged).
    assert augment_prompt_for_angle("hello", "unknown_angle_id") == "hello"


def test_prompt_driven_angle_backbones_share_the_instantid_angle_pack():
    """Each prompt-driven backbone (Qwen-Lightning, FLUX.2-klein, SenseNova) in
    the multi-backbone angle-set picker (sc-2003) MUST advertise the same 11
    canonical angles as the InstantID baseline. The picker depends on every
    backbone exposing the same angle ids so the dropdown shows identical
    angle options regardless of which backbone is selected.
    """
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    instantid_angles = (
        manifest_by_id.get("instantid_realvisxl", {}).get("ui", {}).get("viewAngles") or []
    )
    instantid_ids = {entry["id"] for entry in instantid_angles}
    assert len(instantid_ids) == 11, "Baseline angle pack drifted from 11 canonical angles."
    # The prompt-driven backbones we shipped in the picker matrix:
    for model_id in (
        "qwen_image_edit_2511_lightning",
        "flux2_klein_9b",
        "sensenova_u1_8b_fast",
    ):
        backbone = manifest_by_id.get(model_id) or {}
        view_angles = backbone.get("ui", {}).get("viewAngles") or []
        ids = {entry["id"] for entry in view_angles}
        assert ids == instantid_ids, (
            f"{model_id} advertises viewAngles that don't match InstantID's pack: "
            f"missing={instantid_ids - ids}, extra={ids - instantid_ids}. "
            f"The Character Studio angle dropdown depends on all four backbones "
            f"exposing the same 11 angle ids."
        )


def test_best_effort_backbones_declare_pose_library_capability():
    """sc-2256 / sc-2262: the best-effort pose backbones must advertise
    ui.poseLibrary so the Character Studio pose picker (poseModels filter)
    includes them alongside the strict InstantID backbone."""
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    instantid = manifest_by_id.get("instantid_realvisxl", {})
    assert instantid.get("ui", {}).get("poseLibrary") is True, "Baseline pose backbone drifted."
    for model_id, story in (
        ("qwen_image_edit_2511_lightning", "sc-2256"),
        ("flux2_klein_9b", "sc-2262"),
        # The other two klein variants share the same mlx_flux2 best-effort pose
        # path, so they're offered in the picker for output comparison too.
        ("flux2_klein_9b_kv", "sc-2262"),
        ("flux2_klein_9b_true_v2", "sc-2262"),
    ):
        entry = manifest_by_id.get(model_id, {})
        assert entry.get("ui", {}).get("poseLibrary") is True, (
            f"{model_id} must declare ui.poseLibrary so the pose picker offers its "
            f"best-effort tier ({story})."
        )


def test_qwen_image_adapter_angleset_loop_uses_augmented_prompts(monkeypatch):
    """When advanced.angleSet is set on a character_image request, the
    QwenImageAdapter loops the 11 canonical angles and routes each to
    _run_pipeline with the per-angle prompt_override — NOT the original
    user prompt. Sanity-check this without loading torch / diffusers."""
    import sys as _sys
    from types import ModuleType, SimpleNamespace as _SimpleNamespace

    from scene_worker.character_studio_angles import ANGLE_PROMPT_AUGMENTS, ANGLE_SET_ORDER
    from scene_worker.image_adapters import QwenImageAdapter

    # CI (sceneworks-core-pytest) installs requirements-dev.txt only — torch
    # isn't there. Stub the few attrs the generate() path touches so the test
    # runs against the pure-Python dispatch logic without pulling torch.
    if "torch" not in _sys.modules:
        torch_stub = ModuleType("torch")
        torch_stub.cuda = _SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        torch_stub.backends = _SimpleNamespace(mps=_SimpleNamespace(is_available=lambda: False))
        torch_stub.mps = _SimpleNamespace(empty_cache=lambda: None)
        monkeypatch.setitem(_sys.modules, "torch", torch_stub)

    captured: list[str] = []

    def fake_run_pipeline(
        self, settings, pipe, request, seed, project_path, *, cancel_requested=None, prompt_override=None, pose_skeleton=None
    ):
        from PIL import Image as _Image
        # Angle path carries NO skeleton — it's prompt-driven, not multi-image.
        assert pose_skeleton is None
        captured.append(prompt_override or request.prompt)
        return _Image.new("RGB", (8, 8))

    # Stub out the heavy load + LoRA paths.
    monkeypatch.setattr(QwenImageAdapter, "_run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(QwenImageAdapter, "_load_pipeline", lambda self, *args, **kwargs: object())
    monkeypatch.setattr(QwenImageAdapter, "_apply_loras", lambda self, *args, **kwargs: None)
    # Don't actually write asset files in the test — short-circuit the writer.
    from scene_worker import image_adapters as ia

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, **_kwargs):
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)
    # Bypass the GPU-backend-required check and the device-activation step:
    # both poke at torch internals the stub above doesn't fully model, and the
    # dispatch logic we're verifying doesn't depend on the answer.
    monkeypatch.setattr(ia, "require_inference_backend_for_gpu_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "activate_torch_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "select_torch_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *args, **kwargs: None)

    request = ia.ImageRequest(
        project_id="p",
        mode="character_image",
        prompt="a photo of the character",
        negative_prompt="",
        model="qwen_image_edit_2511_lightning",
        count=1,
        seed=42,
        seeds=[],
        width=1024,
        height=1024,
        style_preset="",
        loras=[],
        character_id=None,
        character_look_id=None,
        source_asset_id=None,
        reference_asset_id="ref-asset-id",
        advanced={"angleSet": True},
        model_manifest_entry={},
    )
    fake_settings = _SimpleNamespace(gpu_id="cpu")
    QwenImageAdapter().generate(
        settings=fake_settings,
        job={"id": "job_x", "payload": {}},
        request=request,
        project_path=None,
        progress=lambda *args, **kwargs: None,
        cancel_requested=lambda: False,
    )
    assert len(captured) == len(ANGLE_SET_ORDER) == 11
    # Each captured prompt should contain the corresponding angle's augment.
    for prompt, angle in zip(captured, ANGLE_SET_ORDER):
        augment = ANGLE_PROMPT_AUGMENTS[angle]
        assert augment in prompt, (
            f"angle '{angle}' didn't reach _run_pipeline as an augmented prompt: "
            f"expected snippet {augment!r} in {prompt!r}"
        )


def test_qwen_image_adapter_pose_loop_renders_skeleton_and_passes_multi_image(monkeypatch):
    """sc-2256: advanced.poses loops the selected library poses, renders each as
    an OpenPose skeleton, and routes it to _run_pipeline as pose_skeleton (so the
    reference branch builds image=[reference, skeleton]) with the pose prompt cue.
    Pose takes precedence over angleSet when both are present."""
    import sys as _sys
    from types import ModuleType, SimpleNamespace as _SimpleNamespace

    from scene_worker.character_studio_angles import POSE_SKELETON_PROMPT
    from scene_worker.image_adapters import QwenImageAdapter

    if "torch" not in _sys.modules:
        torch_stub = ModuleType("torch")
        torch_stub.cuda = _SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        torch_stub.backends = _SimpleNamespace(mps=_SimpleNamespace(is_available=lambda: False))
        torch_stub.mps = _SimpleNamespace(empty_cache=lambda: None)
        monkeypatch.setitem(_sys.modules, "torch", torch_stub)

    captured: list[dict] = []

    def fake_run_pipeline(
        self, settings, pipe, request, seed, project_path, *, cancel_requested=None, prompt_override=None, pose_skeleton=None
    ):
        from PIL import Image as _Image
        captured.append({"prompt_override": prompt_override, "pose_skeleton": pose_skeleton})
        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(QwenImageAdapter, "_run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(QwenImageAdapter, "_load_pipeline", lambda self, *args, **kwargs: object())
    monkeypatch.setattr(QwenImageAdapter, "_apply_loras", lambda self, *args, **kwargs: None)
    from scene_worker import image_adapters as ia

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, **_kwargs):
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)
    monkeypatch.setattr(ia, "require_inference_backend_for_gpu_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "activate_torch_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "select_torch_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *args, **kwargs: None)
    # draw_bodypose needs cv2 (not in the requirements-dev CI venv); the actual
    # render is covered by test_draw_bodypose_renders_colored. Here we only verify
    # dispatch, so stub it to a tiny array Image.fromarray can consume.
    import numpy as _np
    monkeypatch.setattr(ia, "draw_bodypose", lambda w, h, kps: _np.zeros((h, w, 3), dtype=_np.uint8))

    # Two library poses; flat 18-point COCO skeletons (values arbitrary but valid).
    kp = [[0.5, 0.1 + 0.04 * i] for i in range(18)]
    request = ia.ImageRequest(
        project_id="p",
        mode="character_image",
        prompt="a photo of the character",
        negative_prompt="",
        model="qwen_image_edit_2511_lightning",
        count=1,
        seed=42,
        seeds=[],
        width=64,
        height=64,
        style_preset="",
        loras=[],
        character_id=None,
        character_look_id=None,
        source_asset_id=None,
        reference_asset_id="ref-asset-id",
        # angleSet also set to prove pose takes precedence (no angle loop runs).
        advanced={"poses": [{"id": "sit_01", "keypoints": kp}, {"id": "stand_01", "keypoints": kp}], "angleSet": True},
        model_manifest_entry={},
    )
    QwenImageAdapter().generate(
        settings=_SimpleNamespace(gpu_id="cpu"),
        job={"id": "job_pose", "payload": {}},
        request=request,
        project_path=None,
        progress=lambda *args, **kwargs: None,
        cancel_requested=lambda: False,
    )
    # One image per pose (not the 11-angle loop), each with a rendered skeleton
    # and the pose cue appended to the prompt.
    assert len(captured) == 2
    for entry in captured:
        assert entry["pose_skeleton"] is not None
        assert POSE_SKELETON_PROMPT in (entry["prompt_override"] or "")


def test_load_state_dict_requires_weights_only(tmp_path):
    """sc-4230 / F-WORKER-6: .pth upscaler weights load with weights_only=True
    (no pickle execution), and there is NO silent fallback to an unrestricted
    torch.load — an ancient torch without the flag fails loudly instead."""
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"not really a checkpoint")

    class _SafeTorch:
        def load(self, path, map_location=None, weights_only=False):
            assert weights_only is True, "must load with weights_only=True"
            return {"module.body.weight": 1, "other": 2}

    state = _load_state_dict(_SafeTorch(), weights)
    # `module.` prefix stripped; the safe load result is returned.
    assert state == {"body.weight": 1, "other": 2}

    class _AncientTorch:
        """torch.load that predates weights_only — it must NOT be retried unsafely."""

        def load(self, path, map_location=None, **kwargs):
            if "weights_only" in kwargs:
                raise TypeError("load() got an unexpected keyword argument 'weights_only'")
            raise AssertionError("must not fall back to an unrestricted torch.load")

    with pytest.raises(RuntimeError, match="weights_only"):
        _load_state_dict(_AncientTorch(), weights)
