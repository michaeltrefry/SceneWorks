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
    ImageAssetWriter,
    LensTurboAdapter,
    MODEL_TARGETS,
    QwenImageAdapter,
    REAL_ESRGAN_MODEL_SPECS,
    RealEsrganUpscaler,
    ZImageDiffusersAdapter,
    create_image_adapter,
    create_image_upscaler,
    emit_worker_event,
    format_batch_running_message,
    gpu_memory_snapshot,
    huggingface_repo_cache_path,
    image_batch_progress,
    image_request_from_job,
    lens_resolution_for,
    model_supports_edit,
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
    RealESRGANUpscaler,
    TileSlice,
    UpscaleJob,
    create_upscaler_engine,
    tile_slices,
)
from scene_worker.lora_adapters import (
    apply_loras_to_pipeline,
    first_safetensors_path,
    lora_cache_key,
    lora_weight,
    normalize_lora_specs,
    reject_loras_if_unsupported,
    resolve_lora_file,
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
    run_video_job,
    worker_capabilities,
)
from scene_worker.training_adapters import (
    SUPPORTED_LR_SCHEDULERS,
    SUPPORTED_TRAINING_PLAN_VERSION,
    LensLoraTrainer,
    LtxMlxLoraTrainer,
    TrainingKernelError,
    ZImageLoraTrainer,
    _ZImageLoraBackend,
    _build_mlx_lr_schedule,
    _build_mlx_optimizer,
    build_lr_scheduler,
    build_optimizer,
    bucket_resolution,
    create_training_kernel,
    dry_run_training_summary,
    flow_matching_velocity_target,
    lr_decay_multiplier,
    lr_schedule_updates,
    normalize_lr_scheduler,
    read_run_config,
    require_mlx_runtime,
    resolve_pretrained_source,
    resolve_training_adapter_source,
    sample_training_timestep,
    seeded_sample,
    training_adapter_weight_name,
    validate_training_plan,
)
from scene_worker.video_adapters import (
    DiffusersVideoAdapter,
    LtxPipelinesVideoAdapter,
    MlxVideoAdapter,
    VIDEO_MODEL_TARGETS,
    VendorPatchDriftError,
    _PENDING_LTX_LORAS,
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
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu", "nvidia"]})

    # lora_train dry-run validation needs no inference backend, so it is
    # advertised even without torch; generation job types are not.
    assert capabilities == ["gpu", "lora_train", "nvidia"]
    for job_type in ("image_generate", "image_edit", "image_vqa", "video_generate", "training_caption"):
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
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    job_capabilities = [capability for capability in capabilities if capability != "gpu"]

    assert job_capabilities == [
        "image_edit",
        "image_generate",
        "image_interleave",
        "image_vqa",
        "lora_train",
        "lora_train_execute",
        "person_replace",
        "training_caption",
        "video_bridge",
        "video_extend",
        "video_generate",
    ]


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


def test_qwen_loaded_models_track_text_and_edit_repos_independently():
    adapter = QwenImageAdapter()
    adapter._text_repo = "Qwen/Qwen-Image"
    adapter._edit_repo = "Qwen/Qwen-Image-Edit"
    adapter._loaded_model = "qwen_image_edit"

    assert set(adapter.loaded_models()) == {"Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit", "qwen_image_edit"}


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


def _mlx_video_job(model, mode, loras):
    return {"id": "job", "payload": {"projectId": "proj", "model": model, "mode": mode, "loras": loras}}


def test_mlx_adapter_rejects_incompatible_lora_family(tmp_path):
    lora_file = tmp_path / "wan_style.safetensors"
    lora_file.write_bytes(b"lora")
    request = video_request_from_job(
        _mlx_video_job("ltx_2_3", "text_to_video", [{"id": "wan_style", "installedPath": str(lora_file), "family": "wan-video"}])
    )
    with pytest.raises(RuntimeError, match="not compatible with model family ltx-video"):
        MlxVideoAdapter().ensure_models(request)


def test_mlx_condition_image_resolves_source_asset_from_file_path(tmp_path):
    # Regression: MLX image_to_video jobs failed with "Image-to-video mode
    # requires a source image asset." because the adapter read a non-existent
    # top-level `mediaPath` sidecar key instead of the real nested `file.path`,
    # so the source image never resolved. (MLX supports only text/image-to-video;
    # first_last_frame is rejected in ensure_models, so there's no _last frame.)
    project_path = tmp_path / "project"
    (project_path / "assets" / "images").mkdir(parents=True)
    image_rel = "assets/images/source.png"
    Image.new("RGB", (16, 16), "teal").save(project_path / image_rel)
    (project_path / "assets" / "images" / "source.sceneworks.json").write_text(
        json.dumps({"id": "asset-source", "file": {"path": image_rel}}),
        encoding="utf-8",
    )
    adapter = MlxVideoAdapter()

    i2v_request = video_request_from_job(
        {
            "id": "job-i2v",
            "payload": {
                "projectId": "project-1",
                "model": "ltx_2_3",
                "mode": "image_to_video",
                "sourceAssetId": "asset-source",
            },
        }
    )
    first_image = adapter._first_condition_image(project_path, i2v_request)
    assert first_image is not None
    assert first_image.mode == "RGB"
    # Validation now passes instead of raising the source-image error.
    adapter._validate_inputs(project_path, i2v_request, first_image, None)


def test_mlx_wan_user_loras_resolved_to_path_strength_tuples(tmp_path):
    lora_file = tmp_path / "wan_motion.safetensors"
    lora_file.write_bytes(b"lora")
    request = video_request_from_job(
        _mlx_video_job(
            "wan_2_2",
            "text_to_video",
            [{"id": "wan_motion", "installedPath": str(lora_file), "weight": 0.7, "family": "wan-video"}],
        )
    )
    assert MlxVideoAdapter()._wan_user_loras(request) == [(str(lora_file), 0.7)]


def test_mlx_ltx_stages_loras_in_contextvar_only_when_present(tmp_path, monkeypatch):
    lora_file = tmp_path / "ltx_style.safetensors"
    lora_file.write_bytes(b"lora")
    adapter = MlxVideoAdapter()
    monkeypatch.setattr("scene_worker.video_adapters._install_ltx_lora_patch", lambda: None)
    monkeypatch.setattr(adapter, "_ltx_lora_module_map", lambda specs: {"transformer_blocks.0.attn": [("w", specs[0].weight)]})
    progress_calls: list[tuple] = []

    def progress(*args, **kwargs):
        progress_calls.append(args)

    no_lora_request = video_request_from_job(_mlx_video_job("ltx_2_3", "text_to_video", []))
    assert adapter._apply_ltx_loras(no_lora_request, progress) is None
    assert _PENDING_LTX_LORAS.get() is None
    assert progress_calls == []

    lora_request = video_request_from_job(
        _mlx_video_job(
            "ltx_2_3",
            "text_to_video",
            [{"id": "ltx_style", "installedPath": str(lora_file), "weight": 0.6, "family": "ltx-video"}],
        )
    )
    token = adapter._apply_ltx_loras(lora_request, progress)
    try:
        assert token is not None
        assert _PENDING_LTX_LORAS.get() == {"transformer_blocks.0.attn": [("w", 0.6)]}
        assert progress_calls  # merge progress emitted
    finally:
        _PENDING_LTX_LORAS.reset(token)
    assert _PENDING_LTX_LORAS.get() is None


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
        "lora_train",
        "training_caption",
        # sc-1635: VQA + interleave are advertised and dispatched, so the check
        # must report them too.
        "image_vqa",
        "image_interleave",
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


def test_build_mlx_lr_schedule_constant_no_warmup_is_plain_float():
    # Plain constant (no warmup) collapses to a plain float — byte-identical to the
    # pre-scheduler MLX path, with no schedule callable and no MLX dependency.
    assert _build_mlx_lr_schedule("constant", 0.1, total_updates=10, warmup_updates=0) == pytest.approx(0.1)


def test_build_mlx_lr_schedule_callable_returns_mx_array_matching_multiplier():
    mx = pytest.importorskip("mlx.core")

    # Non-constant / warmup returns a callable that mirrors the SAME shared
    # multiplier the torch LambdaLR uses, so both backends decay identically. The
    # value MUST be an mx.array, not a Python float: MLX stores the schedule's
    # return straight into optimizer state and calls ``.astype()`` on it, so a
    # float would crash on the first optimizer update.
    base = 0.1
    for name, total, warmup in [("cosine", 10, 4), ("linear", 8, 0), ("constant", 12, 3)]:
        schedule = _build_mlx_lr_schedule(name, base, total_updates=total, warmup_updates=warmup)
        assert callable(schedule)
        for step in range(total + 1):
            value = schedule(step)
            assert isinstance(value, mx.array)
            assert float(value) == pytest.approx(base * lr_decay_multiplier(name, step, total, warmup))

    # The warmup first step is nonzero (1/(warmup+1) of base), not a wasted 0-LR
    # update — the divergence the torch path deliberately avoids.
    warmup_schedule = _build_mlx_lr_schedule("cosine", base, total_updates=10, warmup_updates=4)
    assert float(warmup_schedule(0)) > 0.0
    assert float(warmup_schedule(0)) == pytest.approx(base * (1 / 5))


def test_build_mlx_lr_schedule_drives_real_optimizer_lr_per_update():
    # Integration guard: feed the schedule to a real mlx.optimizers optimizer
    # (exactly as the LTX MLX backend does) and confirm the effective LR follows
    # lr_decay_multiplier, advancing once per optimizer.update() from the
    # optimizer's own 0-indexed step counter — the same curve the torch LambdaLR
    # path produces. This exercises the float-vs-mx.array contract that the
    # in-isolation callable test cannot.
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    base = 0.05
    total, warmup = 10, 0
    schedule = _build_mlx_lr_schedule("cosine", base, total_updates=total, warmup_updates=warmup)
    optimizer = _build_mlx_optimizer("adamw", schedule, 0.0)
    model = nn.Linear(2, 2)

    observed = []
    for _ in range(total):
        _loss, grads = nn.value_and_grad(model, lambda m: mx.sum(m(mx.ones((1, 2))) ** 2))(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        observed.append(float(optimizer.learning_rate))

    expected = [base * lr_decay_multiplier("cosine", step, total, warmup) for step in range(total)]
    assert observed == pytest.approx(expected, abs=1e-6)
    # And it actually decays — regression guard for the float-return crash that
    # silently left every non-constant MLX schedule non-functional.
    assert observed[0] > observed[total // 2] > observed[-1]


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
                    "sampleGuidanceScale": 0.0,
                    "samplePrompts": ["miraStyle portrait"],
                },
            },
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
    assert calls[0]["guidance_scale"] == 0.0
    assert calls[0]["seed"] == 11
    assert samples[0]["sampleSource"] == "live_adapter"
    assert samples[0]["numInferenceSteps"] == 11
    assert samples[0]["guidanceScale"] == 0.0


def test_create_training_kernel_resolves_known_and_rejects_unknown():
    assert isinstance(create_training_kernel("z_image_lora"), ZImageLoraTrainer)
    assert isinstance(create_training_kernel("lens_lora"), LensLoraTrainer)
    with pytest.raises(TrainingKernelError, match="No training kernel"):
        create_training_kernel("not_a_kernel")


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
                "loraTargetModules": ["img_qkv", "txt_qkv", "to_out", "to_add_out"],
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
        "guidanceScale": 0.0,
        "sampleSource": "live_adapter",
    }
    sample_updates = [payload for payload in progress_results if payload]
    assert sample_updates[-1]["latestTrainingSamples"][0]["relativePath"].startswith("loras/lora_1/samples/")
    assert sample_updates[-1]["sampleSettings"]["guidanceScale"] == 0.0


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


def test_require_mlx_runtime_rejects_non_apple_silicon(monkeypatch):
    import scene_worker.training_adapters as ta

    monkeypatch.setattr(ta.sys, "platform", "linux")
    monkeypatch.setattr(ta.platform, "machine", lambda: "x86_64")
    with pytest.raises(TrainingKernelError, match="Apple Silicon"):
        require_mlx_runtime()


def test_require_mlx_runtime_rejects_apple_silicon_without_mlx(monkeypatch):
    import scene_worker.training_adapters as ta

    monkeypatch.setattr(ta.sys, "platform", "darwin")
    monkeypatch.setattr(ta.platform, "machine", lambda: "arm64")

    def _missing(name):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(ta.importlib, "import_module", _missing)
    with pytest.raises(TrainingKernelError, match="optional MLX worker dependencies"):
        require_mlx_runtime()


def test_require_mlx_runtime_passes_on_apple_silicon_with_mlx(monkeypatch):
    import scene_worker.training_adapters as ta

    monkeypatch.setattr(ta.sys, "platform", "darwin")
    monkeypatch.setattr(ta.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(ta.importlib, "import_module", lambda name: ModuleType(name))
    # Apple Silicon + MLX available: no error.
    require_mlx_runtime()


def test_create_training_kernel_resolves_ltx_mlx():
    kernel = create_training_kernel("ltx_mlx_lora")
    assert isinstance(kernel, LtxMlxLoraTrainer)
    # The LTX trainer reuses the Z-Image backend-agnostic staged orchestration.
    assert isinstance(kernel, ZImageLoraTrainer)


def test_ltx_mlx_trainer_runs_stages_with_fake_backend(tmp_path):
    plan = _real_train_plan(tmp_path, steps=4, save_every=2)
    plan["target"] = {
        "targetId": "ltx_video_lora",
        "kernel": "ltx_mlx_lora",
        "family": "ltx-video",
        "baseModel": "ltx_2_3",
        "baseModelRepo": "notapalindrome/ltx23-mlx-av-q4",
        "baseModelPath": str(tmp_path / "model"),
    }
    backend = FakeTrainingBackend()
    trainer = LtxMlxLoraTrainer(backend=backend)
    events_log = []

    result = trainer.train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda status, stage, value, message, *extra: events_log.append((status, stage)),
        cancel_requested=lambda: False,
    )

    stages = {stage for _status, stage in events_log}
    statuses = {status for status, _stage in events_log}
    assert backend.events[0] == "load"
    assert ("step", 4) in backend.events
    assert result["kernel"] == "ltx_mlx_lora"
    assert result["mode"] == "train"
    assert result["stepsCompleted"] == 4
    assert {"loading_model", "caching_latents", "training", "saving"}.issubset(stages)
    assert statuses <= _VALID_JOB_STATUSES
    assert result["outputPath"] == backend.saved
    assert result["triggerWords"] == ["miraStyle"]


def test_run_lora_train_job_executes_real_run(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None: callback(lambda: False),
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
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None: callback(lambda: False),
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
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None: callback(lambda: False),
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

    def run_immediately(_api, _settings, _job_id, _status, callback, *, loaded_models, on_force_terminate=None):
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


def test_mlx_routing_is_mode_aware_on_mps(monkeypatch):
    # On an MPS host the MLX adapter only handles text_to_video / image_to_video;
    # every other mode must stay on the PyTorch path or it fails in ensure_models.
    monkeypatch.delenv("SCENEWORKS_VIDEO_ADAPTER", raising=False)
    monkeypatch.setattr("scene_worker.video_adapters._mps_available", lambda: True)

    def adapter(model, mode):
        return create_video_adapter({"payload": {"model": model, "mode": mode}}).__class__.__name__

    # Supported MLX modes route to MLX.
    assert adapter("ltx_2_3", "text_to_video") == "MlxVideoAdapter"
    assert adapter("ltx_2_3", "image_to_video") == "MlxVideoAdapter"
    assert adapter("wan_2_2", "image_to_video") == "MlxVideoAdapter"

    # Unsupported modes fall through to the PyTorch adapters, not MLX.
    assert adapter("ltx_2_3", "first_last_frame") == "LtxPipelinesVideoAdapter"
    assert adapter("ltx_2_3", "extend_clip") == "LtxPipelinesVideoAdapter"
    assert adapter("ltx_2_3", "video_bridge") == "LtxPipelinesVideoAdapter"
    assert adapter("wan_2_2", "video_bridge") == "DiffusersVideoAdapter"
    assert adapter("wan_2_2", "replace_person") == "DiffusersVideoAdapter"


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
    # The temp registry now lives on the base VideoGenerationAdapter, so the MLX and
    # Diffusers adapters (which extend the base directly) get force-cancel reaping too
    # via the already-wired on_force_terminate hook (sc-1719).
    adapter = MlxVideoAdapter()
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


def test_mlx_cancel_reaps_temp_but_keeps_pipeline(tmp_path):
    adapter = MlxVideoAdapter()
    adapter._pipeline = object()
    temp = tmp_path / "clip.tmp.mp4"
    temp.write_bytes(b"x")
    adapter.track_temp_output("job-2", temp)

    adapter.cancel("job-2")

    assert not temp.exists()  # temp reaped on cooperative cancel
    assert adapter._pipeline is not None  # ...but the resident pipeline stays loaded


def test_mlx_cleanup_reaps_temp_and_evicts_pipeline(tmp_path):
    adapter = MlxVideoAdapter()
    adapter._pipeline = object()
    temp = tmp_path / "clip.tmp.mp4"
    temp.write_bytes(b"x")
    adapter.track_temp_output("job-3", temp)

    adapter.cleanup("job-3")

    assert not temp.exists()
    assert adapter._pipeline is None


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
    assert list((project_path / "assets" / "videos").glob("*.tmp.mp4"))

    adapter.cleanup(job["id"])

    assert list((project_path / "assets" / "videos").glob("*.tmp.mp4")) == []
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
