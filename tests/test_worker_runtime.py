from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import NamedTuple
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest

from scene_worker.adapter_utils import filter_call_kwargs
from scene_worker.image_adapters import (
    ImageAssetWriter,
    MODEL_TARGETS,
    QwenImageAdapter,
    ZImageDiffusersAdapter,
    build_asset_sidecar,
    create_image_adapter,
    emit_worker_event,
    format_batch_running_message,
    gpu_memory_snapshot,
    huggingface_repo_cache_path,
    image_batch_progress,
    image_request_from_job,
    pipeline_component_devices,
    require_inference_backend_for_gpu_worker,
    resolve_seed,
    select_torch_device,
    verify_pipeline_on_device,
)
from scene_worker.lora_adapters import (
    apply_loras_to_pipeline,
    lora_cache_key,
    lora_weight,
    normalize_lora_specs,
    reject_loras_if_unsupported,
    validate_lora_compatibility,
)
from scene_worker.runtime import (
    child_environment,
    friendly_failure,
    heartbeat,
    keep_job_alive,
    loaded_models_from_adapters,
    main,
    resolve_loaded_models,
    run_check,
    run_video_job,
    worker_capabilities,
)
from scene_worker.video_adapters import (
    DiffusersVideoAdapter,
    LtxPipelinesVideoAdapter,
    VIDEO_MODEL_TARGETS,
    build_video_asset_sidecar,
    character_reference_images,
    create_video_adapter,
    evenly_spaced_indices,
    frames_from_output,
    install_ltx_pipelines_multigpu_compat,
    ltx_model_manifest_entry,
    ltx_frame_count,
    load_seekable_image_frame,
    person_track_masks,
    safe_download_dir,
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
    assert "video_generate" in capabilities
    assert "person_replace" in capabilities
    assert "placeholder" not in capabilities


def test_gpu_worker_without_cuda_torch_does_not_claim_generation_jobs(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu", "nvidia"]})

    assert capabilities == ["gpu", "nvidia"]


def test_python_worker_only_advertises_inference_job_capabilities(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    job_capabilities = [capability for capability in capabilities if capability != "gpu"]

    assert job_capabilities == [
        "image_edit",
        "image_generate",
        "person_replace",
        "video_bridge",
        "video_extend",
        "video_generate",
    ]


def test_python_cpu_worker_does_not_advertise_person_tracking_jobs():
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

    monkeypatch.setattr("scene_worker.runtime.importlib.import_module", lambda name: Torch if name == "torch" else None)

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


def test_huggingface_repo_cache_path_stays_under_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    path = huggingface_repo_cache_path(r"..\outside/../../model")

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
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
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
    assert [len(item["assetIds"]) for item in result_progress] == [1, 2]
    assert result_progress[0]["expectedCount"] == 2
    assert result_progress[0]["generationSetId"] == result["generationSetId"]
    assert result_progress[0]["assets"][0]["file"]["path"].startswith("assets/images/")
    assert result_progress[1]["assetIds"] == result["assetIds"]
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
            assert len(list((project_path / "assets" / "images").glob("*.png"))) == 1
            assert len(list((project_path / "assets" / "images").glob("*.sceneworks.json"))) == 1
        return Image.new("RGB", (16, 16), (255, 0, 0) if index == 0 else (0, 255, 0))

    result = ImageAssetWriter().write_incremental_outputs(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        image_count=2,
        image_at_index=image_at_index,
        adapter_id="z_image_diffusers",
        progress=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
    )

    assert len(result["assetIds"]) == 2
    assert len(list((project_path / "assets" / "images").glob("*.png"))) == 2


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
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
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
    assert "SentencePiece" in error
    assert "pip install -r apps/worker/requirements.txt" in error
    assert "docker compose build worker --no-cache" in error
    assert "Technical detail" in error


def test_worker_check_reports_inference_sidecar_capabilities(monkeypatch):
    events = []
    monkeypatch.setattr("scene_worker.runtime.emit", events.append)
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
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
    ]
    assert events[0]["supportedJobTypes"] == events[0]["jobTypes"]


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

    def run_immediately(_api, _settings, _job_id, _status, callback, *, loaded_models):
        blocking_models.append(loaded_models())
        result = callback()
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
        lambda *_args, **_kwargs: _args[4](),
    )

    run_video_job(
        Api(),
        SimpleNamespace(worker_id="worker-1"),
        {"id": "job-1", "payload": {"projectId": "project-1", "prompt": "clip"}},
    )

    assert "Estimated 121 frames for this clip." in progress_messages


def test_random_batch_seeds_are_used_per_image():
    assert resolve_seed(None, "city at night", 2, [101, 202, 303, 404]) == 303


def test_explicit_seed_uses_reproducible_ladder():
    assert resolve_seed(1234, "city at night", 2, [101, 202, 303, 404]) == 1236


def test_video_adapter_override_aliases_and_unknown_values(monkeypatch):
    monkeypatch.delenv("SCENEWORKS_VIDEO_ADAPTER", raising=False)
    assert create_video_adapter({"payload": {"model": "ltx_2_3"}}).__class__.__name__ == "LtxPipelinesVideoAdapter"
    assert create_video_adapter({"payload": {"model": "wan_2_2"}}).__class__.__name__ == "DiffusersVideoAdapter"
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
    (manifest_dir / "builtin.models.jsonc").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "models": [
                    {
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
                ],
            }
        ),
        encoding="utf-8",
    )


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
    write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
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
    write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)

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
    (manifest_dir / "builtin.models.jsonc").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "models": [
                    {
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
                ],
            }
        ),
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
    write_native_ltx_manifest(config_dir)
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
    write_native_ltx_manifest(config_dir)
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
    write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, gemma=gemma)
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


def test_ltx_model_manifest_entry_reads_jsonc_comments(tmp_path):
    config_dir = tmp_path / "config"
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "builtin.models.jsonc").write_text(
        """
        {
          "schemaVersion": 1,
          "models": [
            {
              // Keep comment stripping out of quoted strings like "https://example.test".
              "id": "ltx_2_3",
              "resources": { "checkpoint": { "path": "models/checkpoint.safetensors" } }
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    entry = ltx_model_manifest_entry(SimpleNamespace(config_dir=config_dir), "ltx_2_3")

    assert entry["resources"]["checkpoint"]["path"] == "models/checkpoint.safetensors"


def test_ltx_model_manifest_entry_preserves_builtin_resources_for_user_entry(tmp_path):
    config_dir = tmp_path / "config"
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "builtin.models.jsonc").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "models": [
                    {
                        "id": "ltx_2_3",
                        "paths": {"model": "data/models/builtin"},
                        "resources": {"checkpoint": {"path": "models/checkpoint.safetensors"}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (manifest_dir / "user.models.jsonc").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "models": [
                    {
                        "id": "ltx_2_3",
                        "paths": {"model": "data/models/user"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    entry = ltx_model_manifest_entry(SimpleNamespace(config_dir=config_dir), "ltx_2_3")

    assert entry["paths"]["model"] == "data/models/user"
    assert entry["resources"]["checkpoint"]["path"] == "models/checkpoint.safetensors"


def test_native_ltx_adapter_rejects_unsupported_modes():
    adapter = LtxPipelinesVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "replace_person",
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

    asset = result["assets"][0]
    media_path = project_path / asset["file"]["path"]
    assert media_path.exists()
    assert result["adapter"] == "ltx_pipelines"
    assert asset["recipe"]["adapter"] == "ltx_pipelines"
    assert asset["recipe"]["rawAdapterSettings"]["pipeline"] == "ltx_pipelines.ti2vid_two_stages"
    assert asset["recipe"]["rawAdapterSettings"]["mockedNativeInference"] is True
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
    write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
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

    asset = result["assets"][0]
    media_path = project_path / asset["file"]["path"]
    assert media_path.read_bytes() == b"mp4"
    assert calls["init"]["checkpoint_path"] == str(checkpoint)
    assert calls["init"]["distilled_lora"] == [(str(lora), 0.6, {"rename": "map"})]
    assert calls["run"]["prompt"] == "Neon harbor"
    assert calls["run"]["negative_prompt"] == "rain"
    assert calls["run"]["num_inference_steps"] == 7
    assert calls["run"]["images"] == []
    assert calls["encode"]["video"] == ["video-chunk"]
    assert calls["encode"]["audio"] == "audio-track"
    assert calls["encode"]["video_chunks_number"] == 2
    assert asset["file"]["mimeType"] == "video/mp4"
    assert asset["recipe"]["rawAdapterSettings"]["realModelInference"] is True
    assert asset["recipe"]["rawAdapterSettings"]["mockedNativeInference"] is False
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
    write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
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
    assert result["assets"][0]["lineage"]["sourceAssetId"] == "asset-source"
    assert result["assets"][0]["recipe"]["rawAdapterSettings"]["realModelInference"] is True


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
    write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
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
    write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
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
    assert result["assets"][0]["lineage"]["sourceClipAssetId"] == "asset-source-video"


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
    write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)

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


def test_character_image_recipe_marks_conditioning_inactive():
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "character_image",
            "prompt": "Mira portrait",
            "model": "z_image_turbo",
            "characterId": "character-1",
            "characterLookId": "look-1",
            "advanced": {},
        },
    }
    request = image_request_from_job(job)

    asset = build_asset_sidecar(
        asset_id="asset-1",
        project_id="project-1",
        generation_set_id="genset-1",
        request=request,
        job_id="job-1",
        media_rel="assets/images/mira.png",
        created_at="2026-05-17T00:00:00Z",
        seed=101,
        index=0,
        model_target=MODEL_TARGETS["z_image_turbo"],
        adapter_id="procedural_preview",
        raw_settings={},
    )

    normalized = asset["recipe"]["normalizedSettings"]
    assert normalized["characterId"] == "character-1"
    assert normalized["characterLookId"] == "look-1"
    assert normalized["characterConditioningActive"] is False


def test_replace_person_video_sidecar_preserves_lineage():
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

    asset = build_video_asset_sidecar(
        asset_id="asset-output",
        project_id="project-1",
        generation_set_id="genset-1",
        request=request,
        job_id="job-1",
        media_rel="assets/videos/replacement.webp",
        created_at="2026-05-17T00:00:00Z",
        seed=44,
        target=VIDEO_MODEL_TARGETS["wan_2_2"],
        adapter_id="wan_video",
        mime_type="video/mp4",
        raw_settings={},
    )

    assert asset["recipe"]["mode"] == "replace_person"
    assert asset["file"]["mimeType"] == "video/mp4"
    assert asset["recipe"]["normalizedSettings"]["personTrackId"] == "track-1"
    assert asset["recipe"]["normalizedSettings"]["replacementMode"] == "full_person_keep_outfit"
    assert asset["recipe"]["normalizedSettings"]["personDetectionActive"] is False
    assert asset["recipe"]["normalizedSettings"]["personTrackingActive"] is False
    assert asset["recipe"]["normalizedSettings"]["replacementActive"] is False
    assert asset["lineage"]["sourceClipAssetId"] == "asset-video"
    assert asset["lineage"]["characterId"] == "character-1"


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
            progress=lambda *_args, **_kwargs: None,
            cancel_requested=lambda: False,
        )
