
# Shared fixtures and helpers for the split worker runtime tests.

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
    RealESRGANUpscaler,
    TileSlice,
    UpscaleJob,
    _load_state_dict,
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

def _run_config_for_network(advanced=None):
    plan = {"config": {"rank": 8, "alpha": 8, "advanced": advanced or {}}}
    return read_run_config(plan)

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

class _FakeStepPipe:
    def __init__(self):
        self._interrupt = False

    def __call__(self, *, prompt=None, callback_on_step_end=None):
        return None

class _NoCallbackPipe:
    def __call__(self, *, prompt=None):
        return None

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

__all__ = [name for name in globals() if not name.startswith("__")]
