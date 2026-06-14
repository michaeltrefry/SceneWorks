"""Narrow Python execution kernels for SceneWorks native LoRA training.

Rust owns the training product surface: dataset storage, manifests, validation,
queue semantics, the target registry, and LoRA registration. A kernel here is a
thin ML runtime that consumes a fully normalized, Rust-resolved ``TrainingPlan``
(see ``crates/sceneworks-core/src/training.rs``) and produces adapter weights. It
never reads SceneWorks storage, config defaults, or the target registry directly.

The first production kernel is :class:`ZImageLoraTrainer`, an image LoRA trainer
for Z-Image-Turbo. Its training mechanics follow the diffusers ``ZImagePipeline``
and ``ostris/ai-toolkit`` as reference material (the epic treats ai-toolkit as a
source of defaults and terminology only): a single-stream DiT transformer trained
with a flow-matching objective whose velocity target is ``noise - latents`` and
whose timestep input is ``(1000 - timestep) / 1000``.

Heavy ML imports (``torch``, ``diffusers``, ``peft``) are deferred to call time so
this module stays importable on a worker without an inference backend — the
dry-run plan validation path needs no backend at all. All orchestration (stage
progress, cancellation, checkpoint cadence, saving) lives in the trainer and is
unit-tested with a fake backend; the model-specific work lives behind a small
backend seam that real GPU runs exercise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import contextlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from sceneworks_shared import utc_now

from .adapter_utils import filter_call_kwargs
from .image_adapters import (
    MODEL_TARGETS,
    activate_torch_device,
    empty_torch_cache,
    emit_worker_event,
    gpu_memory_snapshot,
    require_inference_backend_for_gpu_worker,
    select_torch_device,
    select_torch_dtype,
)
from .lora_adapters import (
    huggingface_main_snapshot_dir,
    huggingface_repo_cache_path,
    huggingface_snapshot_dirs,
)
from .settings import WorkerSettings


ProgressCallback = Callable[[str, str, float, str, dict[str, Any] | None], None]
CancelCallback = Callable[[], bool]

# Highest training plan version a kernel here understands. Plans are resolved in
# Rust (crates/sceneworks-core/src/training.rs::TRAINING_PLAN_VERSION); a kernel
# rejects any version it cannot interpret rather than guessing.
SUPPORTED_TRAINING_PLAN_VERSION = 1

# Default PEFT target modules for the Z-Image single-stream DiT attention blocks.
# PEFT matches by suffix, so these select e.g. ``...attn.to_q`` without needing
# the full module path. Override per job via ``advanced.loraTargetModules``.
DEFAULT_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


class TrainingKernelError(RuntimeError):
    """A training run cannot proceed for a kernel/runtime reason (bad plan,
    missing model component, unsupported diffusers build, ...)."""


# --------------------------------------------------------------------------- #
# Shared plan validation + dry-run summary (used by the dry-run and real paths)
# --------------------------------------------------------------------------- #


def validate_training_plan(plan: Any, *, require_images: bool = True) -> list[dict[str, Any]]:
    """Validate a resolved training plan and return its dataset items.

    Raises ``ValueError`` for a structurally unusable plan and ``FileNotFoundError``
    when dataset images are missing on the worker. Shared by the dry-run validator
    and the real kernel so both reject the same bad inputs identically.
    """

    if not isinstance(plan, dict):
        raise ValueError("Training job payload is missing a resolved plan.")
    plan_version = plan.get("planVersion")
    if plan_version != SUPPORTED_TRAINING_PLAN_VERSION:
        raise ValueError(
            f"Unsupported training plan version {plan_version!r}; this worker "
            f"understands version {SUPPORTED_TRAINING_PLAN_VERSION}."
        )
    dataset = plan.get("dataset") or {}
    items = dataset.get("items") or []
    if not items:
        raise ValueError("Training plan dataset has no items to train on.")
    if require_images:
        missing = [
            item.get("imagePath")
            for item in items
            if not (item.get("imagePath") and os.path.exists(item["imagePath"]))
        ]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} dataset image(s) are missing on the worker, e.g. {preview}."
            )
    return items


def dry_run_training_summary(plan: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Build the dry-run completion summary: what a real run would produce,
    without loading a model or training."""

    dataset = plan.get("dataset") or {}
    items = dataset.get("items") or []
    output = plan.get("output") or {}
    target = plan.get("target") or {}
    base_model_path = target.get("baseModelPath")
    return {
        "mode": "dry_run",
        "validated": True,
        "dryRun": dry_run,
        "datasetItemCount": len(items),
        "datasetId": dataset.get("datasetId"),
        "datasetVersion": dataset.get("datasetVersion"),
        "targetId": target.get("targetId"),
        "kernel": target.get("kernel"),
        "loraId": output.get("loraId"),
        "outputDir": output.get("outputDir"),
        "fileName": output.get("fileName"),
        "baseModel": target.get("baseModel"),
        "baseModelRepo": target.get("baseModelRepo"),
        "baseModelPath": base_model_path,
        "baseModelInstalled": bool(base_model_path and os.path.exists(base_model_path)),
        "planVersion": plan.get("planVersion"),
        "completedAt": utc_now(),
    }


# --------------------------------------------------------------------------- #
# Resolved config + plan accessors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainingRunConfig:
    """Concrete hyperparameters read from a plan's ``config`` block. Mirrors the
    Rust ``TrainingConfig`` shape but is duck-typed from the plan dict so the
    kernel never depends on the Rust crate."""

    rank: int
    alpha: int
    learning_rate: float
    steps: int
    batch_size: int
    gradient_accumulation: int
    resolution: int
    save_every: int
    seed: int
    optimizer: str
    weight_decay: float
    timestep_type: str
    timestep_bias: str
    loss_type: str
    # Learning-rate scheduler (distinct from the flow-matching noise scheduler
    # configured by ``timestep_type``/``timestep_bias``). ``constant`` holds the
    # optimizer LR fixed; ``linear``/``cosine`` decay it over the run, after an
    # optional ``lr_warmup_steps`` linear ramp.
    lr_scheduler: str
    lr_warmup_steps: int
    gradient_checkpointing: bool
    mixed_precision: Any
    lora_target_modules: Any
    sample_every: int
    sample_steps: int
    sample_guidance_scale: float
    sample_prompts: list[str]
    training_adapter_repo: str | None = None
    training_adapter_version: str | None = None
    weight_noise_sigma: float = 0.0
    # Adapter network parameterization. ``lora`` is the default; ``lokr`` builds a
    # LyCORIS Kronecker-product adapter (peft.LoKrConfig). ``decompose_factor`` is
    # the LoKr block-split knob (``-1`` = auto). Both live in the free-form
    # ``advanced`` bag in the contract (gated per target by ``limits.networkTypes``).
    network_type: str = "lora"
    decompose_factor: int = -1
    advanced: dict[str, Any] = field(default_factory=dict)


def _as_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def read_run_config(plan: dict[str, Any]) -> TrainingRunConfig:
    config = plan.get("config") or {}
    advanced = config.get("advanced") if isinstance(config.get("advanced"), dict) else {}
    target_modules = advanced.get("loraTargetModules")
    if not target_modules:
        target_modules = list(DEFAULT_LORA_TARGET_MODULES)
    sample_prompts = advanced.get("samplePrompts")
    if not isinstance(sample_prompts, list):
        sample_prompts = default_sample_prompts(trigger_words(plan))
    return TrainingRunConfig(
        rank=_as_int(config.get("rank"), 16, minimum=1),
        alpha=_as_int(config.get("alpha"), 16, minimum=1),
        learning_rate=_as_float(config.get("learningRate"), 1e-4),
        steps=_as_int(config.get("steps"), 1000, minimum=1),
        batch_size=_as_int(config.get("batchSize"), 1, minimum=1),
        gradient_accumulation=_as_int(config.get("gradientAccumulation"), 1, minimum=1),
        resolution=_as_int(config.get("resolution"), 1024, minimum=32),
        save_every=_as_int(config.get("saveEvery"), 0, minimum=0),
        seed=_as_int(config.get("seed"), 42),
        optimizer=str(config.get("optimizer") or "adamw"),
        weight_decay=_as_float(advanced.get("weightDecay"), 0.0),
        timestep_type=str(advanced.get("timestepType") or "sigmoid"),
        timestep_bias=str(advanced.get("timestepBias") or "balanced"),
        loss_type=str(advanced.get("lossType") or "mse"),
        lr_scheduler=str(advanced.get("lrScheduler") or "constant"),
        lr_warmup_steps=_as_int(advanced.get("lrWarmupSteps"), 0, minimum=0),
        gradient_checkpointing=_as_bool(advanced.get("gradientCheckpointing"), True),
        mixed_precision=advanced.get("mixedPrecision"),
        lora_target_modules=target_modules,
        sample_every=_as_int(advanced.get("sampleEvery"), 0, minimum=0),
        sample_steps=_as_int(advanced.get("sampleSteps"), 9, minimum=1),
        sample_guidance_scale=_as_float(
            advanced.get("sampleGuidanceScale"),
            default_sample_guidance_scale(plan),
        ),
        sample_prompts=[str(prompt).strip() for prompt in sample_prompts if str(prompt).strip()][:4],
        training_adapter_repo=_as_optional_str(advanced.get("trainingAdapterRepo")),
        training_adapter_version=_as_optional_str(advanced.get("trainingAdapterVersion")),
        weight_noise_sigma=max(0.0, _as_float(advanced.get("weightNoiseSigma"), 0.0)),
        network_type=str(advanced.get("networkType") or "lora").strip().lower(),
        decompose_factor=_as_int(advanced.get("decomposeFactor"), -1, minimum=-1),
        advanced=advanced,
    )


def default_sample_guidance_scale(plan: dict[str, Any]) -> float:
    target = plan.get("target") if isinstance(plan.get("target"), dict) else {}
    if target.get("kernel") == "z_image_lora" or target.get("baseModel") in {"z_image_turbo", "z_image_edit"}:
        return float(MODEL_TARGETS["z_image_turbo"].get("guidanceScale", 1.0))
    return 0.0


def trigger_words(plan: dict[str, Any]) -> list[str]:
    output = plan.get("output") or {}
    words = output.get("triggerWords") or []
    return [str(word) for word in words if str(word).strip()]


def default_sample_prompts(words: list[str]) -> list[str]:
    trigger = ", ".join(words).strip() or "the trained subject"
    return [
        f"{trigger}, studio portrait, soft key light, detailed face",
        f"{trigger}, full body fashion editorial photo, natural pose",
        f"{trigger}, cinematic outdoor portrait, golden hour",
        f"{trigger}, close-up character portrait, dramatic rim light",
    ]


def project_relative_path(plan: dict[str, Any], path: Path) -> str | None:
    dataset_root = Path(str((plan.get("dataset") or {}).get("rootPath") or ""))
    try:
        # Dataset roots are <project>/training/datasets/<dataset-id>.
        project_root = dataset_root.parents[2]
        relative = path.resolve().relative_to(project_root.resolve())
    except Exception:
        return None
    return relative.as_posix()


# --------------------------------------------------------------------------- #
# Base-model source resolution
# --------------------------------------------------------------------------- #


def resolve_pretrained_source(target: dict[str, Any]) -> str:
    """Resolve a ``from_pretrained`` source for the base model.

    Prefers a directly loadable directory (a SceneWorks-managed import or an
    explicit snapshot containing ``model_index.json``), then the main snapshot of
    a Hugging Face cache repo root, then the repo id (so diffusers can use the
    cache or download), then the raw path as a last resort.
    """

    repo = str(target.get("baseModelRepo") or "").strip()
    base_path = str(target.get("baseModelPath") or "").strip()
    if base_path:
        path = Path(base_path)
        if (path / "model_index.json").is_file():
            return str(path)
        snapshot = _snapshot_with_model_index(path)
        if snapshot is not None:
            return str(snapshot)
    if repo:
        return repo
    if base_path:
        return base_path
    raise TrainingKernelError("Training plan has no base model repo or path to load.")


def _snapshot_with_model_index(repo_root: Path) -> Path | None:
    """Return the cache snapshot dir under an HF ``models--*`` root that holds a
    ``model_index.json``, preferring the ``refs/main`` snapshot."""

    main_snapshot = huggingface_main_snapshot_dir(repo_root)
    if main_snapshot is not None and (main_snapshot / "model_index.json").is_file():
        return main_snapshot
    for snapshot in huggingface_snapshot_dirs(repo_root):
        if (snapshot / "model_index.json").is_file():
            return snapshot
    return None


# --------------------------------------------------------------------------- #
# Z-Image-Turbo de-distill training adapter
# --------------------------------------------------------------------------- #

# Z-Image-Turbo is a step-distilled model. Training a LoRA directly on it makes
# the distillation break down unpredictably (blurry, off-identity output even as
# the loss optimizes), so ostris ships a "de-distill" training adapter: it is
# fused into the transformer for training only, the new LoRA is learned on top,
# and the adapter is dropped at inference (left on the plain distilled model).
# See https://huggingface.co/ostris/zimage_turbo_training_adapter. The repo holds
# exactly the two weight files below; presets pick one via ``trainingAdapterVersion``.
TRAINING_ADAPTER_WEIGHT_FILES = {
    "v1": "zimage_turbo_training_adapter_v1.safetensors",
    "v2": "zimage_turbo_training_adapter_v2.safetensors",
}


def training_adapter_weight_name(version: str | None) -> str:
    """Map a ``trainingAdapterVersion`` (e.g. ``v1``, ``v2-default``) to the repo's
    weight file. Defaults to v2, the SceneWorks preset default."""

    token = (version or "").strip().lower()
    if "v1" in token:
        return TRAINING_ADAPTER_WEIGHT_FILES["v1"]
    return TRAINING_ADAPTER_WEIGHT_FILES["v2"]


def resolve_training_adapter_source(repo: str, version: str | None) -> tuple[str, str]:
    """Resolve ``(load_target, weight_name)`` for the de-distill adapter.

    Prefers a locally cached weight file under the HF hub cache (returned as an
    absolute path so ``load_lora_weights`` needs no network); otherwise returns the
    repo id so diffusers can use its own cache or download it, matching how the base
    model source is resolved."""

    weight_name = training_adapter_weight_name(version)
    root = huggingface_repo_cache_path(repo)
    if root is not None and root.exists():
        for snapshot in huggingface_snapshot_dirs(root):
            candidate = snapshot / weight_name
            if candidate.is_file():
                return str(candidate), weight_name
    return repo, weight_name


# --------------------------------------------------------------------------- #
# Backend seam
# --------------------------------------------------------------------------- #


class TrainingBackend(Protocol):
    """The model-specific work behind the trainer's orchestration. The real
    backend wraps torch/diffusers/peft; tests pass a fake implementation."""

    def loaded_models(self) -> list[str]: ...

    def load(
        self,
        *,
        settings: WorkerSettings,
        plan: dict[str, Any],
        config: TrainingRunConfig,
        progress: ProgressCallback,
    ) -> None: ...

    def prepare_dataset(
        self,
        *,
        items: list[dict[str, Any]],
        config: TrainingRunConfig,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]: ...

    def train_step(self, *, step: int, total_steps: int, config: TrainingRunConfig) -> float: ...

    def save_checkpoint(self, *, step: int, output_dir: str, file_name: str) -> str | None: ...

    def generate_samples(
        self,
        *,
        step: int,
        prompts: list[str],
        output_dir: str,
        file_name: str,
        plan: dict[str, Any],
        config: TrainingRunConfig,
    ) -> list[dict[str, Any]]: ...

    def save_final(self, *, output_dir: str, file_name: str) -> str: ...

    def cleanup(self) -> None: ...


# Progress band: preparing 0.0-0.08, loading 0.08-0.18, caching 0.18-0.32,
# training 0.32-0.92, saving 0.92-0.98, completed (posted by the runtime) 1.0.
_TRAIN_PROGRESS_START = 0.32
_TRAIN_PROGRESS_END = 0.92
_CACHE_PROGRESS_START = 0.18
_CACHE_PROGRESS_END = 0.32
# Report a training-progress tick at most this often (in steps) to avoid flooding
# the API on long runs; the final step always reports.
PROGRESS_STEP_INTERVAL = 10


def _scaled(start: float, end: float, completed: int, total: int) -> float:
    safe_total = max(1, total)
    fraction = min(max(0, completed), safe_total) / safe_total
    return round(start + (end - start) * fraction, 4)


def _check_cancel(cancel_requested: CancelCallback) -> None:
    if cancel_requested():
        raise InterruptedError("LoRA training canceled by user.")


class ZImageLoraTrainer:
    """Image LoRA trainer for Z-Image-Turbo.

    Orchestrates the staged run (prepare → load → cache → train → checkpoint →
    save) with cancellation and progress reporting. The model-specific work is
    delegated to a :class:`TrainingBackend`; the real one is built lazily so the
    trainer is importable and unit-testable without an inference backend.
    """

    kernel_id = "z_image_lora"

    def __init__(self, backend: TrainingBackend | None = None) -> None:
        self._backend = backend
        self._active_backend: TrainingBackend | None = backend

    def loaded_models(self) -> list[str]:
        backend = self._active_backend
        if backend is None:
            return []
        try:
            return list(backend.loaded_models())
        except Exception:
            return []

    def train(
        self,
        *,
        settings: WorkerSettings,
        plan: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        items = validate_training_plan(plan, require_images=True)
        config = read_run_config(plan)
        output = plan.get("output") or {}
        target = plan.get("target") or {}
        output_dir = str(output.get("outputDir") or "")
        file_name = str(output.get("fileName") or "lora.safetensors")
        if not output_dir:
            raise TrainingKernelError("Training plan output is missing an output directory.")

        progress("preparing", "preparing", 0.04, "Preparing LoRA training run.")
        backend = self._backend or self._create_backend()
        self._active_backend = backend
        completed_steps = 0
        try:
            progress(
                "loading_model",
                "loading_model",
                0.1,
                f"Loading base model for {target.get('targetId') or self.kernel_id}.",
            )
            backend.load(settings=settings, plan=plan, config=config, progress=progress)
            _check_cancel(cancel_requested)

            progress(
                "running",
                "caching_latents",
                _CACHE_PROGRESS_START,
                f"Encoding {len(items)} dataset item(s).",
            )
            prepared = backend.prepare_dataset(
                items=items,
                config=config,
                progress=progress,
                cancel_requested=cancel_requested,
            )
            _check_cancel(cancel_requested)

            total_steps = max(1, config.steps)
            checkpoints: list[dict[str, Any]] = []
            training_samples: list[dict[str, Any]] = []
            for step in range(1, total_steps + 1):
                _check_cancel(cancel_requested)
                loss = backend.train_step(step=step, total_steps=total_steps, config=config)
                completed_steps = step
                if step == total_steps or step % PROGRESS_STEP_INTERVAL == 0:
                    progress(
                        "running",
                        "training",
                        _scaled(_TRAIN_PROGRESS_START, _TRAIN_PROGRESS_END, step, total_steps),
                        _training_message(step, total_steps, loss),
                    )
                if config.save_every and step % config.save_every == 0 and step < total_steps:
                    progress(
                        "running",
                        "checkpointing",
                        _scaled(_TRAIN_PROGRESS_START, _TRAIN_PROGRESS_END, step, total_steps),
                        f"Saving checkpoint at step {step} of {total_steps}.",
                    )
                    checkpoint_path = backend.save_checkpoint(
                        step=step, output_dir=output_dir, file_name=file_name
                    )
                    if checkpoint_path:
                        checkpoints.append({"step": step, "path": checkpoint_path})
                if config.sample_every and step % config.sample_every == 0:
                    progress(
                        "running",
                        "rendering",
                        _scaled(_TRAIN_PROGRESS_START, _TRAIN_PROGRESS_END, step, total_steps),
                        f"Rendering training samples at step {step} of {total_steps}.",
                    )
                    samples = backend.generate_samples(
                        step=step,
                        prompts=config.sample_prompts,
                        output_dir=output_dir,
                        file_name=file_name,
                        plan=plan,
                        config=config,
                    )
                    if samples:
                        training_samples.extend(samples)
                        progress(
                            "running",
                            "rendering",
                            _scaled(_TRAIN_PROGRESS_START, _TRAIN_PROGRESS_END, step, total_steps),
                            f"Rendered {len(samples)} training sample(s) at step {step}.",
                            {
                                "trainingSamples": training_samples,
                                "latestTrainingSamples": samples,
                                "samplePrompts": config.sample_prompts,
                                "sampleSettings": {
                                    "numInferenceSteps": config.sample_steps,
                                    "guidanceScale": config.sample_guidance_scale,
                                    "sampleSource": "live_adapter",
                                },
                            },
                        )

            _check_cancel(cancel_requested)
            progress("saving", "saving", 0.95, "Saving trained LoRA weights.")
            output_path = backend.save_final(output_dir=output_dir, file_name=file_name)
            return self._result_summary(
                plan=plan,
                config=config,
                prepared=prepared,
                total_steps=total_steps,
                completed_steps=completed_steps,
                checkpoints=checkpoints,
                training_samples=training_samples,
                output_path=output_path,
            )
        finally:
            try:
                backend.cleanup()
            except Exception:
                pass

    def _create_backend(self) -> TrainingBackend:
        return _ZImageLoraBackend()

    def _result_summary(
        self,
        *,
        plan: dict[str, Any],
        config: TrainingRunConfig,
        prepared: dict[str, Any],
        total_steps: int,
        completed_steps: int,
        checkpoints: list[dict[str, Any]],
        training_samples: list[dict[str, Any]],
        output_path: str,
    ) -> dict[str, Any]:
        dataset = plan.get("dataset") or {}
        target = plan.get("target") or {}
        output = plan.get("output") or {}
        return {
            "mode": "train",
            "kernel": self.kernel_id,
            "loraId": output.get("loraId"),
            "outputDir": output.get("outputDir"),
            "fileName": output.get("fileName"),
            "outputPath": output_path,
            "format": output.get("format") or "safetensors",
            "datasetId": dataset.get("datasetId"),
            "datasetVersion": dataset.get("datasetVersion"),
            "datasetItemCount": (prepared or {}).get("itemCount") or len(dataset.get("items") or []),
            "targetId": target.get("targetId"),
            "baseModel": target.get("baseModel"),
            "steps": total_steps,
            "stepsCompleted": completed_steps,
            "checkpoints": checkpoints,
            "trainingSamples": training_samples,
            "latestTrainingSamples": training_samples[-4:],
            "samplePrompts": config.sample_prompts,
            "sampleSettings": {
                "numInferenceSteps": config.sample_steps,
                "guidanceScale": config.sample_guidance_scale,
                "sampleSource": "live_adapter",
            },
            "rank": config.rank,
            "alpha": config.alpha,
            "learningRate": config.learning_rate,
            "lrScheduler": config.lr_scheduler,
            "lrWarmupSteps": config.lr_warmup_steps,
            "resolution": (prepared or {}).get("resolution") or config.resolution,
            "triggerWords": trigger_words(plan),
            "planVersion": plan.get("planVersion"),
            "completedAt": utc_now(),
        }


def _training_message(step: int, total_steps: int, loss: float | None) -> str:
    if loss is None:
        return f"Training step {step} of {total_steps}."
    return f"Training step {step} of {total_steps} (loss {loss:.4f})."


# --------------------------------------------------------------------------- #
# Real torch/diffusers/peft backend for Z-Image
# --------------------------------------------------------------------------- #


def build_optimizer(name: str, params: list[Any], learning_rate: float, weight_decay: float = 0.0) -> Any:
    """Build an optimizer for the LoRA parameters. ``adamw8bit`` uses
    bitsandbytes when available and falls back to torch AdamW otherwise (the
    8-bit optimizer is an optional, CUDA-only dependency). ``prodigy`` and
    ``prodigyopt`` use the Prodigy optimizer package. ``rose`` uses the rose-opt
    package (stateless range-normalized updates, drop-in for AdamW)."""

    torch = importlib.import_module("torch")
    normalized = (name or "").strip().lower().replace("-", "").replace("_", "")
    if normalized in {"prodigy", "prodigyopt"}:
        try:
            prodigy_module = importlib.import_module("prodigyopt")
        except Exception as exc:
            raise TrainingKernelError("The Prodigy optimizer requires the prodigyopt Python package.") from exc
        use_lr = learning_rate if learning_rate >= 0.1 else 1.0
        return prodigy_module.Prodigy(params, lr=use_lr, eps=1e-6, weight_decay=weight_decay)
    if normalized in {"rose", "roseopt"}:
        try:
            rose_module = importlib.import_module("rose_opt")
        except Exception as exc:
            raise TrainingKernelError("The Rose optimizer requires the rose-opt Python package.") from exc
        # compute_dtype="fp32": Rose's fp64 default has no MPS kernel; fp32 is
        # safe on both CUDA and Apple Silicon.
        return rose_module.Rose(params, lr=learning_rate, weight_decay=weight_decay, compute_dtype="fp32")
    if normalized in {"adamw8bit", "adam8bit"}:
        try:
            bnb = importlib.import_module("bitsandbytes")
            return bnb.optim.AdamW8bit(params, lr=learning_rate, weight_decay=weight_decay)
        except Exception:
            emit_worker_event(
                "training_optimizer_fallback",
                requested=name,
                using="adamw",
                reason="bitsandbytes unavailable",
            )
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def build_peft_network_config(peft: Any, config: TrainingRunConfig) -> Any:
    """Build the PEFT adapter config for the requested network type.

    ``lora`` (the default) yields a standard low-rank ``LoraConfig``; ``lokr``
    yields a LyCORIS Kronecker-product ``LoKrConfig`` (epic 2193). Both attach via
    the same ``model.add_adapter(...)`` seam and save via ``get_peft_model_state_dict``
    — only the on-disk key layout and inference loader differ. Per-target gating
    (which backends accept ``lokr``) lives in the Rust contract's
    ``limits.networkTypes``; this is the kernel-side constructor."""

    target_modules = (
        list(config.lora_target_modules)
        if isinstance(config.lora_target_modules, (list, tuple))
        else config.lora_target_modules
    )
    if (config.network_type or "lora").strip().lower() == "lokr":
        return peft.LoKrConfig(
            r=config.rank,
            alpha=config.alpha,
            decompose_factor=config.decompose_factor,
            init_weights=True,
            target_modules=target_modules,
        )
    return peft.LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )


def write_lokr_adapter(
    state_dict: dict[str, Any],
    output_dir: str,
    file_name: str,
    *,
    rank: int,
    alpha: int,
    decompose_factor: int,
    target_modules: Any,
) -> str:
    """Write a LoKr adapter to safetensors with routing + reconstruction metadata.

    diffusers' ``save_lora_weights``/``load_lora_weights`` only understand LoRA
    (``lora_A``/``lora_B``) keys — LoKr emits ``lokr_w1``/``lokr_w2``/``lokr_t2``,
    so the trainer serializes the raw ``get_peft_model_state_dict`` directly. The
    file header carries everything the inference loader (epic 2193) needs to
    rebuild the matching ``peft.LoKrConfig`` and inject it (``load_lora_weights``
    cannot consume LoKr): ``networkType`` to route, plus ``rank``/``alpha``/
    ``decomposeFactor``/``targetModules`` to reconstruct the network. safetensors
    metadata is ``str``→``str``, so ints are stringified and the module list is
    JSON-encoded."""

    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, file_name)
    tensors = {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}
    modules = list(target_modules) if isinstance(target_modules, (list, tuple)) else target_modules
    metadata = {
        "format": "pt",
        "networkType": "lokr",
        "rank": str(int(rank)),
        "alpha": str(int(alpha)),
        "decomposeFactor": str(int(decompose_factor)),
        "targetModules": json.dumps(modules),
    }
    save_file(tensors, path, metadata=metadata)
    return path


# Learning-rate schedulers both backends honor. The flow-matching *noise*
# scheduler (sigmoid/linear/weighted timestep sampling) is a separate concept,
# configured via ``timestepType``/``timestepBias`` — see ``sample_training_timestep``.
SUPPORTED_LR_SCHEDULERS = ("constant", "linear", "cosine")


def normalize_lr_scheduler(name: str | None) -> str:
    """Normalize an ``lrScheduler`` name to a supported value, raising a clear
    error for anything outside :data:`SUPPORTED_LR_SCHEDULERS`. Rust validates the
    same set at submit time; this is the kernel-side backstop for plans handed to
    the worker directly."""

    normalized = (name or "constant").strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_LR_SCHEDULERS:
        raise TrainingKernelError(
            f"Unsupported lrScheduler '{name}'. Supported schedulers: "
            + ", ".join(SUPPORTED_LR_SCHEDULERS)
            + "."
        )
    return normalized


def lr_decay_multiplier(name: str, step: int, total: int, warmup: int) -> float:
    """Base-LR multiplier in [0, 1] at optimizer-update ``step`` (0-indexed),
    shared by the torch and MLX schedule builders so both kernels decay
    identically. An optional linear warmup ramps to 1.0 over ``warmup`` updates
    (no dead 0.0 first step), then the body decays: ``linear`` to 0, ``cosine`` on
    a half-cosine to 0, ``constant`` holds at 1.0."""

    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup + 1)
    if total <= warmup:
        return 1.0
    progress = min(1.0, max(0.0, float(step - warmup) / float(total - warmup)))
    if name == "linear":
        return 1.0 - progress
    if name == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0  # constant (with warmup)


def lr_schedule_updates(steps: int, gradient_accumulation: int, warmup_steps: int) -> tuple[int, int]:
    """Convert micro-step counts to optimizer-update counts (the scheduler steps
    once per optimizer update, which gradient accumulation makes less frequent
    than micro-steps). Returns ``(total_updates, warmup_updates)`` with warmup
    clamped below the run so the body always has room to decay."""

    accum = max(1, int(gradient_accumulation))
    total = max(1, (max(1, int(steps)) + accum - 1) // accum)
    warmup = (max(0, int(warmup_steps)) + accum - 1) // accum
    return total, max(0, min(warmup, total - 1))


def build_lr_scheduler(
    torch: Any,
    optimizer: Any,
    name: str | None,
    *,
    total_updates: int,
    warmup_updates: int,
) -> Any | None:
    """Build a ``torch.optim.lr_scheduler.LambdaLR`` that scales each param
    group's base LR by :func:`lr_decay_multiplier`, stepped once per optimizer
    update. Returns ``None`` for plain ``constant`` (no warmup) so the optimizer
    LR stays exactly fixed — byte-identical to every pre-scheduler run. Raises
    ``TrainingKernelError`` for an unsupported scheduler name."""

    normalized = normalize_lr_scheduler(name)
    total = max(1, int(total_updates))
    warmup = max(0, min(int(warmup_updates), total - 1))
    if normalized == "constant" and warmup == 0:
        return None

    def lr_lambda(step: int) -> float:
        return lr_decay_multiplier(normalized, step, total, warmup)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def seeded_sample(torch: Any, fn: Any, shape: Any, *, generator: Any, device: Any, dtype: Any) -> Any:
    """Draw seeded random values, MPS-safe.

    ``torch.Generator`` only lives on cpu/cuda, so on Apple Silicon a seeded run
    pairs a cpu generator with tensors on ``mps``. ``torch.randn`` / ``torch.rand``
    reject a cpu generator alongside a non-cpu ``device=`` argument, so when the
    generator's device differs from the target device we draw on the generator's
    device and move — mirroring diffusers' ``randn_tensor``. ``fn`` is
    ``torch.randn`` or ``torch.rand``.
    """
    if generator is not None and generator.device.type != torch.device(device).type:
        return fn(shape, generator=generator, device=generator.device, dtype=dtype).to(device)
    return fn(shape, generator=generator, device=device, dtype=dtype)


def sample_training_timestep(
    torch: Any,
    *,
    generator: Any,
    device: str,
    dtype: Any,
    timestep_type: str,
    timestep_bias: str,
) -> Any:
    """Sample a normalized flow-matching timestep in [0, 1].

    The `sigmoid` shape follows ai-toolkit's flowmatch scheduler: random normal
    values are passed through sigmoid so most samples land near the middle of the
    denoising range. Bias then nudges the normalized value toward high or low
    noise while keeping the same single-sample training loop.
    """

    normalized_type = (timestep_type or "sigmoid").strip().lower().replace("-", "_")
    if normalized_type in {"linear", "uniform"}:
        t = seeded_sample(torch, torch.rand, 1, generator=generator, device=device, dtype=dtype)
    elif normalized_type == "weighted":
        base = seeded_sample(torch, torch.rand, 1, generator=generator, device=device, dtype=dtype)
        center = torch.sigmoid(
            seeded_sample(torch, torch.randn, 1, generator=generator, device=device, dtype=dtype)
        )
        t = (base + center) / 2.0
    else:
        t = torch.sigmoid(
            seeded_sample(torch, torch.randn, 1, generator=generator, device=device, dtype=dtype)
        )

    normalized_bias = (timestep_bias or "balanced").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_bias in {"high", "high_noise", "favor_high_noise"}:
        t = torch.sqrt(t)
    elif normalized_bias in {"low", "low_noise", "favor_low_noise"}:
        t = t * t
    return t.clamp(1e-3, 1.0 - 1e-3)


def training_loss(torch: Any, prediction: Any, target: Any, loss_type: str) -> Any:
    normalized = (loss_type or "mse").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"mae", "l1", "mean_absolute_error"}:
        return torch.nn.functional.l1_loss(prediction.float(), target.float())
    return torch.nn.functional.mse_loss(prediction.float(), target.float())


def apply_weight_noise(torch: Any, optimizer: Any, sigma: float) -> None:
    """Add small Gaussian noise to every trainable parameter the optimizer owns.

    Implements the "weight noising" regularizer from BuffaloBuffaloBuffaloBuffalo /
    ai-toolkit-perceptual: a per-step Gaussian perturbation that biases LoRA
    training toward flatter loss minima and reduces character-LoRA memorization.
    Opt-in via ``advanced.weightNoiseSigma`` (``0`` disables — the default).
    """

    if not sigma or sigma <= 0.0:
        return
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group.get("params", ()):
                if param is None or not param.requires_grad:
                    continue
                param.add_(torch.randn_like(param) * sigma)


def flow_matching_velocity_target(latents: Any, noise: Any) -> Any:
    """Training target for the RAW Z-Image transformer output: ``latents - noise``.

    This is the **negated** flow-matching velocity, and the sign is load-bearing.
    diffusers' ``ZImagePipeline`` negates the transformer output before handing it
    to ``FlowMatchEulerDiscreteScheduler.step`` (``noise_pred = -noise_pred``), and
    that scheduler integrates its input as the velocity ``noise - latents`` (for
    ``x_sigma = (1 - sigma) * latents + sigma * noise``). So the raw output the
    transformer is trained to produce is ``-(noise - latents) = latents - noise``.
    Regressing toward ``noise - latents`` would train the LoRA to push the model in
    the opposite denoising direction while the loss still looks like it converges.

    Works on torch tensors (operator overloading) and plain numbers (for tests).
    """
    return latents - noise


def bucket_resolution(resolution: int) -> int:
    """Floor a resolution to a multiple of 32 (Z-Image VAE factor 16 × patch 2),
    with a sane minimum."""

    bucket = (max(32, int(resolution)) // 32) * 32
    return max(32, bucket)


def _load_training_image(image_path: str, resolution: int) -> Any:
    from PIL import Image

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    # Center-crop to a square, then resize to the bucket edge so aspect ratio is
    # preserved without distortion. Simple and deterministic for a first kernel;
    # aspect-ratio bucketing can come later via advanced settings.
    width, height = image.size
    edge = min(width, height)
    left = (width - edge) // 2
    top = (height - edge) // 2
    square = image.crop((left, top, left + edge, top + edge))
    return square.resize((resolution, resolution), Image.LANCZOS)


def _image_to_tensor(torch: Any, image: Any, dtype: Any, device: str) -> Any:
    import numpy as np

    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


class _ZImageLoraBackend:
    """Real Z-Image LoRA training backend.

    Loads the diffusers ``ZImagePipeline`` components, attaches a PEFT LoRA to the
    transformer, caches per-item latents and prompt embeddings, and runs a
    flow-matching training loop. The forward/loss in :meth:`train_step` follows
    the pipeline's own transformer call and ai-toolkit's velocity target; it is
    deliberately isolated so a real GPU run can be tuned without touching the
    trainer's orchestration.
    """

    def __init__(self) -> None:
        self._torch: Any | None = None
        self._device: str | None = None
        self._dtype: Any | None = None
        self._pipeline: Any | None = None
        self._transformer: Any | None = None
        self._vae: Any | None = None
        self._optimizer: Any | None = None
        self._lr_scheduler: Any | None = None
        self._generator: Any | None = None
        self._loaded_source: str | None = None
        self._latents: list[Any] = []
        self._embeds: list[Any] = []
        self._vae_scaling: float = 1.0
        self._vae_shift: float = 0.0
        self._diagnosed_forward = False

    def loaded_models(self) -> list[str]:
        return [self._loaded_source] if self._loaded_source else []

    def load(
        self,
        *,
        settings: WorkerSettings,
        plan: dict[str, Any],
        config: TrainingRunConfig,
        progress: ProgressCallback,
    ) -> None:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        peft = importlib.import_module("peft")
        self._torch = torch

        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, config.mixed_precision)
        source = resolve_pretrained_source(plan.get("target") or {})

        pipeline_class = getattr(diffusers, "ZImagePipeline", None)
        if pipeline_class is None:
            raise TrainingKernelError(
                "The installed diffusers build does not expose ZImagePipeline; "
                "install a diffusers build with Z-Image support."
            )

        emit_worker_event(
            "training_pipeline_load_start",
            kernel=ZImageLoraTrainer.kernel_id,
            source=source,
            device=device,
            dtype=str(dtype),
        )
        progress("loading_model", "loading_model", 0.12, "Loading Z-Image base model files.")
        pipe = pipeline_class.from_pretrained(source, torch_dtype=dtype)
        pipe.to(device)

        # Z-Image-Turbo is step-distilled; fuse the de-distill training adapter into
        # the base before attaching the trainable LoRA. This learns on a de-distilled
        # model and ships only the new LoRA, which inference applies to the plain
        # distilled model (the adapter is intentionally not saved).
        self._apply_training_adapter(pipe, config, progress)

        transformer = pipe.transformer
        transformer.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None:
            text_encoder.requires_grad_(False)

        progress("loading_model", "loading_model", 0.16, "Attaching LoRA adapter to the transformer.")
        self._network_type = config.network_type
        # Stashed for the save path: everything write_lokr_adapter needs to stamp
        # into the file so inference can rebuild the matching LoKrConfig.
        self._lokr_save_kwargs = {
            "rank": config.rank,
            "alpha": config.alpha,
            "decompose_factor": config.decompose_factor,
            "target_modules": config.lora_target_modules,
        }
        lora_config = build_peft_network_config(peft, config)
        transformer.add_adapter(lora_config)
        self._activate_lora_adapter(transformer)
        if config.gradient_checkpointing:
            # With a frozen base + LoRA, reentrant gradient checkpointing can drop
            # gradients to the adapter (its inputs don't require grad). Force the
            # inputs to require grad so the LoRA actually trains regardless of the
            # transformer's checkpoint implementation.
            if hasattr(transformer, "enable_input_require_grads"):
                try:
                    transformer.enable_input_require_grads()
                except Exception:
                    pass
            if hasattr(transformer, "enable_gradient_checkpointing"):
                transformer.enable_gradient_checkpointing()
            elif hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable()
            else:
                emit_worker_event(
                    "training_gradient_checkpointing_unavailable",
                    kernel=ZImageLoraTrainer.kernel_id,
                    transformer=type(transformer).__name__,
                )
        transformer.train()
        trainable = [param for param in transformer.parameters() if param.requires_grad]
        if not trainable:
            raise TrainingKernelError(
                "LoRA adapter attached no trainable parameters; the configured "
                "target modules did not match any transformer layers. Adjust "
                "advanced.loraTargetModules for this base model."
            )

        self._optimizer = build_optimizer(
            config.optimizer, trainable, config.learning_rate, config.weight_decay
        )
        self._optimizer.zero_grad()
        # Learning-rate scheduler steps once per optimizer update; ``constant``
        # with no warmup yields ``None`` so the LR stays exactly fixed.
        total_updates, warmup_updates = lr_schedule_updates(
            config.steps, config.gradient_accumulation, config.lr_warmup_steps
        )
        self._lr_scheduler = build_lr_scheduler(
            torch,
            self._optimizer,
            config.lr_scheduler,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
        )
        vae_config = getattr(pipe.vae, "config", None)
        self._vae_scaling = float(getattr(vae_config, "scaling_factor", 1.0) or 1.0)
        self._vae_shift = float(getattr(vae_config, "shift_factor", 0.0) or 0.0)
        self._pipeline = pipe
        self._transformer = transformer
        self._vae = pipe.vae
        self._device = device
        self._dtype = dtype
        self._loaded_source = source
        generator_device = device if str(device).startswith("cuda") else "cpu"
        self._generator = torch.Generator(generator_device).manual_seed(int(config.seed))
        lora_a_norm, lora_b_norm = self._lora_param_norms()
        emit_worker_event(
            "training_pipeline_load_complete",
            kernel=ZImageLoraTrainer.kernel_id,
            source=source,
            trainableTensors=len(trainable),
            # Baseline LoRA norms: lora_B starts ~0 (zero-init), so a growing
            # ``loraBNorm`` between here and save proves the adapter is learning.
            loraANorm=lora_a_norm,
            loraBNorm=lora_b_norm,
            lrScheduler=config.lr_scheduler,
            lrWarmupSteps=config.lr_warmup_steps,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

    def _apply_training_adapter(
        self, pipe: Any, config: TrainingRunConfig, progress: ProgressCallback
    ) -> str | None:
        """Fuse the Z-Image-Turbo de-distill adapter into the base transformer.

        Loads the adapter as a LoRA, fuses it into the weights, then unloads the
        (now-redundant) LoRA modules so the de-distill delta lives in the frozen
        base. The trainable LoRA attached afterwards therefore learns on top of a
        de-distilled model, and ``get_peft_model_state_dict`` at save time captures
        only that trainable LoRA — never the fused adapter."""

        repo = (config.training_adapter_repo or "").strip()
        if not repo:
            emit_worker_event(
                "training_dedistill_adapter_skipped",
                kernel=ZImageLoraTrainer.kernel_id,
                reason="no trainingAdapterRepo configured",
            )
            return None
        if not hasattr(pipe, "load_lora_weights") or not hasattr(pipe, "fuse_lora"):
            raise TrainingKernelError(
                "The installed diffusers build cannot load/fuse the Z-Image-Turbo "
                "de-distill training adapter (load_lora_weights/fuse_lora missing)."
            )

        load_target, weight_name = resolve_training_adapter_source(
            repo, config.training_adapter_version
        )
        progress(
            "loading_model",
            "loading_model",
            0.14,
            f"Applying Z-Image-Turbo de-distill adapter ({weight_name}).",
        )
        emit_worker_event(
            "training_dedistill_adapter_load_start",
            kernel=ZImageLoraTrainer.kernel_id,
            repo=repo,
            version=config.training_adapter_version,
            weightName=weight_name,
            source=load_target,
        )
        try:
            if os.path.exists(load_target):
                pipe.load_lora_weights(load_target, adapter_name="dedistill")
            else:
                pipe.load_lora_weights(
                    load_target, weight_name=weight_name, adapter_name="dedistill"
                )
            pipe.fuse_lora()
            if hasattr(pipe, "unload_lora_weights"):
                pipe.unload_lora_weights()
        except Exception as exc:
            raise TrainingKernelError(
                "Failed to apply the Z-Image-Turbo de-distill training adapter "
                f"({repo}/{weight_name}). Z-Image-Turbo is step-distilled, and "
                "training without this adapter produces unusable LoRAs. "
                f"Underlying error: {exc}"
            ) from exc
        emit_worker_event(
            "training_dedistill_adapter_applied",
            kernel=ZImageLoraTrainer.kernel_id,
            repo=repo,
            weightName=weight_name,
        )
        return weight_name

    def _lora_param_norms(self) -> tuple[float, float]:
        """Return ``(lora_A_norm, lora_B_norm)`` over the trainable adapter params.

        Cheap diagnostic so a flat vs growing ``lora_B`` norm (it starts ~0) is
        visible in worker events without hand-parsing the saved safetensors."""

        transformer = self._transformer
        if transformer is None or not hasattr(transformer, "named_parameters"):
            return 0.0, 0.0
        a_sq = 0.0
        b_sq = 0.0
        for name, param in transformer.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            try:
                value = float(param.detach().float().pow(2).sum().to("cpu"))
            except Exception:
                continue
            if "lora_B" in name or "lora_b" in name:
                b_sq += value
            elif "lora_A" in name or "lora_a" in name:
                a_sq += value
        return round(a_sq**0.5, 6), round(b_sq**0.5, 6)

    def prepare_dataset(
        self,
        *,
        items: list[dict[str, Any]],
        config: TrainingRunConfig,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        torch = self._torch
        pipe = self._pipeline
        vae = self._vae
        resolution = bucket_resolution(config.resolution)
        count = len(items)
        self._latents = []
        self._embeds = []
        with torch.no_grad():
            for index, item in enumerate(items):
                if cancel_requested():
                    raise InterruptedError("LoRA training canceled by user.")
                image = _load_training_image(item["imagePath"], resolution)
                pixel = _image_to_tensor(torch, image, self._dtype, self._device)
                latent = vae.encode(pixel).latent_dist.sample(generator=self._generator)
                latent = (latent - self._vae_shift) * self._vae_scaling
                self._latents.append(latent.detach().to("cpu"))

                prompt_embeds, _ = pipe.encode_prompt(
                    str(item.get("caption") or ""),
                    device=self._device,
                    do_classifier_free_guidance=False,
                )
                embed = prompt_embeds[0] if isinstance(prompt_embeds, (list, tuple)) else prompt_embeds
                self._embeds.append(embed.detach().to("cpu"))

                if (index + 1) % 4 == 0 or index + 1 == count:
                    progress(
                        "running",
                        "caching_latents",
                        _scaled(_CACHE_PROGRESS_START, _CACHE_PROGRESS_END, index + 1, count),
                        f"Encoded {index + 1} of {count} dataset item(s).",
                    )
        return {"itemCount": count, "resolution": resolution}

    def train_step(self, *, step: int, total_steps: int, config: TrainingRunConfig) -> float:
        torch = self._torch
        transformer = self._transformer
        device = self._device
        index = (step - 1) % len(self._latents)
        latents = self._latents[index].to(device)
        embeds = self._embeds[index].to(device)

        # Rectified-flow / flow-matching training: interpolate between the clean
        # latent (t=0) and noise (t=1). The transformer takes the timestep scaled
        # as ``(1000 - timestep) / 1000`` and per-item latent/embed lists,
        # mirroring ZImagePipeline.__call__. The target sign matches the diffusers
        # pipeline, which negates the raw transformer output before the scheduler
        # (see flow_matching_velocity_target).
        noise = seeded_sample(
            torch, torch.randn, latents.shape, generator=self._generator, device=device, dtype=latents.dtype
        )
        t = sample_training_timestep(
            torch,
            generator=self._generator,
            device=device,
            dtype=latents.dtype,
            timestep_type=config.timestep_type,
            timestep_bias=config.timestep_bias,
        )
        noisy = (1.0 - t) * latents + t * noise
        target = flow_matching_velocity_target(latents, noise)
        timestep = t * 1000.0
        timestep_model_input = (1000.0 - timestep) / 1000.0

        latent_model_input_list = list(noisy.unsqueeze(2).unbind(dim=0))
        model_out = transformer(
            latent_model_input_list, timestep_model_input, [embeds], return_dict=False
        )[0]
        prediction = self._stack_model_output(torch, model_out)
        if prediction.dim() == target.dim() + 1:
            prediction = prediction.squeeze(2)
        if not self._diagnosed_forward:
            emit_worker_event(
                "training_forward_shapes",
                kernel=ZImageLoraTrainer.kernel_id,
                latent=list(latents.shape),
                prediction=list(prediction.shape),
                target=list(target.shape),
            )
            self._diagnosed_forward = True

        loss = training_loss(torch, prediction, target, config.loss_type)
        accum = max(1, config.gradient_accumulation)
        (loss / accum).backward()
        if step % accum == 0 or step == total_steps:
            self._optimizer.step()
            apply_weight_noise(torch, self._optimizer, config.weight_noise_sigma)
            self._optimizer.zero_grad()
            # Advance the LR scheduler once per optimizer update (``None`` for a
            # plain constant schedule, leaving the LR fixed).
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
        return float(loss.detach().to("cpu"))

    def _stack_model_output(self, torch: Any, model_out: Any) -> Any:
        if isinstance(model_out, (list, tuple)):
            return torch.stack(list(model_out), dim=0)
        return model_out

    def save_checkpoint(self, *, step: int, output_dir: str, file_name: str) -> str | None:
        stem = Path(file_name).stem or "lora"
        checkpoint_name = f"{stem}-step{step:06d}.safetensors"
        return self._save_lora(output_dir=output_dir, file_name=checkpoint_name)

    def generate_samples(
        self,
        *,
        step: int,
        prompts: list[str],
        output_dir: str,
        file_name: str,
        plan: dict[str, Any],
        config: TrainingRunConfig,
    ) -> list[dict[str, Any]]:
        torch = self._torch
        pipe = self._pipeline
        transformer = self._transformer
        if torch is None or pipe is None or transformer is None or not prompts:
            return []

        sample_dir = Path(output_dir) / "samples" / f"step-{step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_name).stem or "lora"
        was_training = bool(getattr(transformer, "training", False))
        self._activate_lora_adapter(transformer)
        transformer.eval()
        samples: list[dict[str, Any]] = []
        try:
            with torch.no_grad():
                for index, prompt in enumerate(prompts[:4]):
                    generator_device = self._device if str(self._device).startswith("cuda") else "cpu"
                    generator = torch.Generator(generator_device).manual_seed(int(config.seed) + step + index)
                    kwargs = {
                        "prompt": prompt,
                        "height": min(768, bucket_resolution(config.resolution)),
                        "width": min(768, bucket_resolution(config.resolution)),
                        "num_inference_steps": config.sample_steps,
                        "guidance_scale": config.sample_guidance_scale,
                        "generator": generator,
                    }
                    output = pipe(**filter_call_kwargs(pipe, kwargs))
                    image = output.images[0].convert("RGB")
                    sample_name = f"{stem}-step{step:06d}-{index + 1}.png"
                    sample_path = sample_dir / sample_name
                    image.save(sample_path)
                    samples.append(
                        {
                            "step": step,
                            "prompt": prompt,
                            "path": str(sample_path),
                            "relativePath": project_relative_path(plan, sample_path),
                            "sampleSource": "live_adapter",
                            "numInferenceSteps": config.sample_steps,
                            "guidanceScale": config.sample_guidance_scale,
                            "createdAt": utc_now(),
                        }
                    )
        finally:
            if was_training:
                transformer.train()
        return samples

    def _activate_lora_adapter(self, transformer: Any) -> None:
        for method_name in ("set_adapter", "enable_adapters"):
            method = getattr(transformer, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "set_adapter":
                    method("default")
                else:
                    method()
                emit_worker_event(
                    "training_lora_adapter_active",
                    kernel=ZImageLoraTrainer.kernel_id,
                    method=method_name,
                )
                return
            except Exception as exc:
                emit_worker_event(
                    "training_lora_adapter_activation_failed",
                    kernel=ZImageLoraTrainer.kernel_id,
                    method=method_name,
                    error=str(exc),
                )

    def save_final(self, *, output_dir: str, file_name: str) -> str:
        return self._save_lora(output_dir=output_dir, file_name=file_name)

    def _save_lora(self, *, output_dir: str, file_name: str) -> str:
        from peft.utils import get_peft_model_state_dict

        os.makedirs(output_dir, exist_ok=True)
        lora_state_dict = get_peft_model_state_dict(self._transformer)
        if getattr(self, "_network_type", "lora") == "lokr":
            # LoKr keys (lokr_w1/lokr_w2) are not save_lora_weights-compatible;
            # write raw with routing metadata (epic 2193).
            write_lokr_adapter(
                lora_state_dict,
                output_dir,
                file_name,
                **getattr(self, "_lokr_save_kwargs", {}),
            )
        else:
            type(self._pipeline).save_lora_weights(
                output_dir,
                transformer_lora_layers=lora_state_dict,
                weight_name=file_name,
                safe_serialization=True,
            )
        lora_a_norm, lora_b_norm = self._lora_param_norms()
        emit_worker_event(
            "training_lora_weight_norm",
            kernel=ZImageLoraTrainer.kernel_id,
            fileName=file_name,
            tensors=len(lora_state_dict),
            loraANorm=lora_a_norm,
            loraBNorm=lora_b_norm,
        )
        return os.path.join(output_dir, file_name)

    def cleanup(self) -> None:
        torch = self._torch
        self._latents = []
        self._embeds = []
        self._optimizer = None
        self._transformer = None
        self._vae = None
        self._pipeline = None
        self._loaded_source = None
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# SDXL-UNet LoRA backend (torch / diffusers / PEFT) — the shared foundation for
# every SDXL-architecture LoRA target. Kolors (epic 1929) subclasses it by
# swapping the pipeline class + the prompt encoder (see the seams below).
# --------------------------------------------------------------------------- #


class _SdxlLoraBackend:
    """Real SDXL-UNet LoRA training backend.

    Loads ``StableDiffusionXLPipeline``, caches per-item VAE latents + frozen
    dual-CLIP prompt embeddings (text + pooled), attaches a PEFT LoRA to
    ``pipe.unet``, and runs the SDXL epsilon/v-prediction objective with the
    SDXL ``added_cond_kwargs`` (pooled text embeds + ``add_time_ids``) on a DDPM
    noise schedule. This is the **first U-Net (non-DiT) trainer** in the repo;
    the existing kernels are flow-matching transformer trainers, so the noise
    schedule, the integer timesteps, and the ``added_cond_kwargs`` forward are
    deliberately isolated here from the shared orchestration.

    Extension seams for epic 1929 (Kolors = SDXL UNet + ChatGLM3): override
    :attr:`pipeline_class_name` and :meth:`_encode_prompt`. Everything else —
    the UNet LoRA injection, the training loop, the save path — is shared.
    """

    # Seams a same-architecture subclass (Kolors) overrides.
    kernel_id = "sdxl_lora"
    pipeline_class_name = "StableDiffusionXLPipeline"
    # SDXL base ships an fp16 variant; Kolors-diffusers is also fp16-only.
    load_variant: str | None = "fp16"

    def __init__(self) -> None:
        self._torch: Any | None = None
        self._device: str | None = None
        self._dtype: Any | None = None
        self._pipeline: Any | None = None
        self._unet: Any | None = None
        self._vae: Any | None = None
        self._noise_scheduler: Any | None = None
        self._num_train_timesteps: int = 1000
        self._prediction_type: str = "epsilon"
        self._optimizer: Any | None = None
        self._lr_scheduler: Any | None = None
        self._generator: Any | None = None
        self._loaded_source: str | None = None
        self._latents: list[Any] = []
        self._prompt_embeds: list[Any] = []
        self._pooled_embeds: list[Any] = []
        self._add_time_ids: Any | None = None
        self._vae_scaling: float = 1.0
        self._diagnosed_forward = False

    def loaded_models(self) -> list[str]:
        return [self._loaded_source] if self._loaded_source else []

    def load(
        self,
        *,
        settings: WorkerSettings,
        plan: dict[str, Any],
        config: TrainingRunConfig,
        progress: ProgressCallback,
    ) -> None:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        peft = importlib.import_module("peft")
        self._torch = torch

        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, config.mixed_precision)
        source = resolve_pretrained_source(plan.get("target") or {})

        pipeline_class = getattr(diffusers, self.pipeline_class_name, None)
        if pipeline_class is None:
            raise TrainingKernelError(
                f"The installed diffusers build does not expose {self.pipeline_class_name}; "
                "install a diffusers build with SDXL support."
            )
        ddpm_class = getattr(diffusers, "DDPMScheduler", None)
        if ddpm_class is None:
            raise TrainingKernelError(
                "The installed diffusers build does not expose DDPMScheduler, "
                "required for the SDXL training noise schedule."
            )

        emit_worker_event(
            "training_pipeline_load_start",
            kernel=self.kernel_id,
            source=source,
            device=device,
            dtype=str(dtype),
        )
        progress("loading_model", "loading_model", 0.12, "Loading SDXL base model files.")
        from_pretrained_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.load_variant:
            from_pretrained_kwargs["variant"] = self.load_variant
        pipe = pipeline_class.from_pretrained(source, **from_pretrained_kwargs)
        pipe.to(device)
        # The SDXL fp16 VAE emits NaN latents; it is frozen and only used for the
        # one-time latent cache, so upcast it to fp32 for numerically safe encodes.
        pipe.vae.to(dtype=torch.float32)

        unet = pipe.unet
        unet.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        for encoder_attr in ("text_encoder", "text_encoder_2"):
            encoder = getattr(pipe, encoder_attr, None)
            if encoder is not None:
                encoder.requires_grad_(False)

        # Train on a DDPM schedule derived from the base model's own scheduler
        # config so num_train_timesteps + prediction_type match (SDXL base is
        # epsilon). The inference scheduler (Euler) is a different object.
        self._noise_scheduler = ddpm_class.from_config(pipe.scheduler.config)
        self._num_train_timesteps = int(
            getattr(self._noise_scheduler.config, "num_train_timesteps", 1000) or 1000
        )
        self._prediction_type = str(
            getattr(self._noise_scheduler.config, "prediction_type", "epsilon") or "epsilon"
        )

        progress("loading_model", "loading_model", 0.16, "Attaching LoRA adapter to the U-Net.")
        self._network_type = config.network_type
        # Stashed for the save path: everything write_lokr_adapter needs to stamp
        # into the file so inference can rebuild the matching LoKrConfig.
        self._lokr_save_kwargs = {
            "rank": config.rank,
            "alpha": config.alpha,
            "decompose_factor": config.decompose_factor,
            "target_modules": config.lora_target_modules,
        }
        lora_config = build_peft_network_config(peft, config)
        unet.add_adapter(lora_config)
        self._activate_lora_adapter(unet)
        if config.gradient_checkpointing:
            # Frozen base + LoRA: force inputs to require grad so reentrant
            # checkpointing doesn't drop gradients to the adapter.
            if hasattr(unet, "enable_input_require_grads"):
                try:
                    unet.enable_input_require_grads()
                except Exception:
                    pass
            if hasattr(unet, "enable_gradient_checkpointing"):
                unet.enable_gradient_checkpointing()
            elif hasattr(unet, "gradient_checkpointing_enable"):
                unet.gradient_checkpointing_enable()
            else:
                emit_worker_event(
                    "training_gradient_checkpointing_unavailable",
                    kernel=self.kernel_id,
                    unet=type(unet).__name__,
                )
        unet.train()
        trainable = [param for param in unet.parameters() if param.requires_grad]
        if not trainable:
            raise TrainingKernelError(
                "LoRA adapter attached no trainable parameters; the configured "
                "target modules did not match any U-Net layers. Adjust "
                "advanced.loraTargetModules for this base model."
            )

        self._optimizer = build_optimizer(
            config.optimizer, trainable, config.learning_rate, config.weight_decay
        )
        self._optimizer.zero_grad()
        total_updates, warmup_updates = lr_schedule_updates(
            config.steps, config.gradient_accumulation, config.lr_warmup_steps
        )
        self._lr_scheduler = build_lr_scheduler(
            torch,
            self._optimizer,
            config.lr_scheduler,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
        )
        vae_config = getattr(pipe.vae, "config", None)
        self._vae_scaling = float(getattr(vae_config, "scaling_factor", 1.0) or 1.0)
        self._pipeline = pipe
        self._unet = unet
        self._vae = pipe.vae
        self._device = device
        self._dtype = dtype
        self._loaded_source = source
        generator_device = device if str(device).startswith("cuda") else "cpu"
        self._generator = torch.Generator(generator_device).manual_seed(int(config.seed))
        lora_a_norm, lora_b_norm = self._lora_param_norms()
        emit_worker_event(
            "training_pipeline_load_complete",
            kernel=self.kernel_id,
            source=source,
            trainableTensors=len(trainable),
            predictionType=self._prediction_type,
            loraANorm=lora_a_norm,
            loraBNorm=lora_b_norm,
            lrScheduler=config.lr_scheduler,
            lrWarmupSteps=config.lr_warmup_steps,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

    def _encode_prompt(self, pipe: Any, caption: str, device: str) -> tuple[Any, Any]:
        """Return ``(prompt_embeds, pooled_prompt_embeds)`` for one caption.

        Overridable seam: SDXL uses the dual-CLIP ``encode_prompt``; epic 1929's
        Kolors backend swaps in ``KolorsPipeline.encode_prompt`` (ChatGLM3,
        ``max_sequence_length=256``) here and changes nothing else.
        """
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=caption,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return prompt_embeds, pooled_prompt_embeds

    def _lora_param_norms(self) -> tuple[float, float]:
        unet = self._unet
        if unet is None or not hasattr(unet, "named_parameters"):
            return 0.0, 0.0
        a_sq = 0.0
        b_sq = 0.0
        for name, param in unet.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            try:
                value = float(param.detach().float().pow(2).sum().to("cpu"))
            except Exception:
                continue
            if "lora_B" in name or "lora_b" in name:
                b_sq += value
            elif "lora_A" in name or "lora_a" in name:
                a_sq += value
        return round(a_sq**0.5, 6), round(b_sq**0.5, 6)

    def _activate_lora_adapter(self, unet: Any) -> None:
        for method_name in ("set_adapter", "enable_adapters"):
            method = getattr(unet, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "set_adapter":
                    method("default")
                else:
                    method()
                return
            except Exception as exc:
                emit_worker_event(
                    "training_lora_adapter_activation_failed",
                    kernel=self.kernel_id,
                    method=method_name,
                    error=str(exc),
                )

    def prepare_dataset(
        self,
        *,
        items: list[dict[str, Any]],
        config: TrainingRunConfig,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        torch = self._torch
        pipe = self._pipeline
        vae = self._vae
        resolution = bucket_resolution(config.resolution)
        count = len(items)
        self._latents = []
        self._prompt_embeds = []
        self._pooled_embeds = []
        # SDXL micro-conditioning: [orig_h, orig_w, crop_top, crop_left,
        # target_h, target_w]. We square center-crop, so original == target and
        # the crop offset is 0. Built once and shared across the (bsz=1) loop.
        self._add_time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            dtype=self._dtype,
            device=self._device,
        )
        with torch.no_grad():
            for index, item in enumerate(items):
                if cancel_requested():
                    raise InterruptedError("LoRA training canceled by user.")
                image = _load_training_image(item["imagePath"], resolution)
                # Encode in fp32 (VAE was upcast in load) to avoid fp16 NaN latents.
                pixel = _image_to_tensor(torch, image, torch.float32, self._device)
                latent = vae.encode(pixel).latent_dist.sample(generator=self._generator)
                latent = latent * self._vae_scaling
                self._latents.append(latent.detach().to("cpu"))

                prompt_embeds, pooled = self._encode_prompt(
                    pipe, str(item.get("caption") or ""), self._device
                )
                self._prompt_embeds.append(prompt_embeds.detach().to("cpu"))
                self._pooled_embeds.append(pooled.detach().to("cpu"))

                if (index + 1) % 4 == 0 or index + 1 == count:
                    progress(
                        "running",
                        "caching_latents",
                        _scaled(_CACHE_PROGRESS_START, _CACHE_PROGRESS_END, index + 1, count),
                        f"Encoded {index + 1} of {count} dataset item(s).",
                    )
        return {"itemCount": count, "resolution": resolution}

    def train_step(self, *, step: int, total_steps: int, config: TrainingRunConfig) -> float:
        torch = self._torch
        unet = self._unet
        device = self._device
        index = (step - 1) % len(self._latents)
        latents = self._latents[index].to(device=device, dtype=self._dtype)
        prompt_embeds = self._prompt_embeds[index].to(device=device, dtype=self._dtype)
        pooled = self._pooled_embeds[index].to(device=device, dtype=self._dtype)

        noise = seeded_sample(
            torch, torch.randn, latents.shape, generator=self._generator, device=device, dtype=latents.dtype
        )
        # SDXL trains on the discrete DDPM schedule: integer timesteps + add_noise,
        # not the flow-matching interpolation the DiT kernels use. The cpu generator
        # (MPS-safe) draws on its own device, then we move to the compute device.
        generator_device = self._generator.device
        timesteps = torch.randint(
            0,
            self._num_train_timesteps,
            (latents.shape[0],),
            generator=self._generator,
            device=generator_device,
        ).to(device)
        noisy = self._noise_scheduler.add_noise(latents, noise, timesteps)
        added_cond_kwargs = {"text_embeds": pooled, "time_ids": self._add_time_ids}
        model_pred = unet(
            noisy,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        if self._prediction_type == "v_prediction":
            target = self._noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        if not self._diagnosed_forward:
            emit_worker_event(
                "training_forward_shapes",
                kernel=self.kernel_id,
                latent=list(latents.shape),
                prediction=list(model_pred.shape),
                target=list(target.shape),
                predictionType=self._prediction_type,
            )
            self._diagnosed_forward = True

        loss = training_loss(torch, model_pred, target, config.loss_type)
        accum = max(1, config.gradient_accumulation)
        (loss / accum).backward()
        if step % accum == 0 or step == total_steps:
            self._optimizer.step()
            apply_weight_noise(torch, self._optimizer, config.weight_noise_sigma)
            self._optimizer.zero_grad()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
        return float(loss.detach().to("cpu"))

    def save_checkpoint(self, *, step: int, output_dir: str, file_name: str) -> str | None:
        stem = Path(file_name).stem or "lora"
        checkpoint_name = f"{stem}-step{step:06d}.safetensors"
        return self._save_lora(output_dir=output_dir, file_name=checkpoint_name)

    def generate_samples(
        self,
        *,
        step: int,
        prompts: list[str],
        output_dir: str,
        file_name: str,
        plan: dict[str, Any],
        config: TrainingRunConfig,
    ) -> list[dict[str, Any]]:
        torch = self._torch
        pipe = self._pipeline
        unet = self._unet
        if torch is None or pipe is None or unet is None or not prompts:
            return []

        sample_dir = Path(output_dir) / "samples" / f"step-{step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_name).stem or "lora"
        was_training = bool(getattr(unet, "training", False))
        self._activate_lora_adapter(unet)
        unet.eval()
        edge = min(1024, bucket_resolution(config.resolution))
        samples: list[dict[str, Any]] = []
        try:
            with torch.no_grad():
                for index, prompt in enumerate(prompts[:4]):
                    generator_device = self._device if str(self._device).startswith("cuda") else "cpu"
                    generator = torch.Generator(generator_device).manual_seed(int(config.seed) + step + index)
                    kwargs = {
                        "prompt": prompt,
                        "height": edge,
                        "width": edge,
                        "num_inference_steps": config.sample_steps,
                        "guidance_scale": config.sample_guidance_scale,
                        "generator": generator,
                    }
                    output = pipe(**filter_call_kwargs(pipe, kwargs))
                    image = output.images[0].convert("RGB")
                    sample_name = f"{stem}-step{step:06d}-{index + 1}.png"
                    sample_path = sample_dir / sample_name
                    image.save(sample_path)
                    samples.append(
                        {
                            "step": step,
                            "prompt": prompt,
                            "path": str(sample_path),
                            "relativePath": project_relative_path(plan, sample_path),
                            "sampleSource": "live_adapter",
                            "numInferenceSteps": config.sample_steps,
                            "guidanceScale": config.sample_guidance_scale,
                            "createdAt": utc_now(),
                        }
                    )
        finally:
            if was_training:
                unet.train()
        return samples

    def save_final(self, *, output_dir: str, file_name: str) -> str:
        return self._save_lora(output_dir=output_dir, file_name=file_name)

    def _save_lora(self, *, output_dir: str, file_name: str) -> str:
        from peft.utils import get_peft_model_state_dict

        os.makedirs(output_dir, exist_ok=True)
        lora_state_dict = get_peft_model_state_dict(self._unet)
        if getattr(self, "_network_type", "lora") == "lokr":
            # LoKr keys (lokr_w1/lokr_w2) are not save_lora_weights-compatible;
            # write raw with routing metadata (epic 2193).
            write_lokr_adapter(
                lora_state_dict,
                output_dir,
                file_name,
                **getattr(self, "_lokr_save_kwargs", {}),
            )
        else:
            type(self._pipeline).save_lora_weights(
                output_dir,
                unet_lora_layers=lora_state_dict,
                weight_name=file_name,
                safe_serialization=True,
            )
        lora_a_norm, lora_b_norm = self._lora_param_norms()
        emit_worker_event(
            "training_lora_weight_norm",
            kernel=self.kernel_id,
            fileName=file_name,
            tensors=len(lora_state_dict),
            loraANorm=lora_a_norm,
            loraBNorm=lora_b_norm,
        )
        return os.path.join(output_dir, file_name)

    def cleanup(self) -> None:
        torch = self._torch
        self._latents = []
        self._prompt_embeds = []
        self._pooled_embeds = []
        self._add_time_ids = None
        self._optimizer = None
        self._unet = None
        self._vae = None
        self._pipeline = None
        self._noise_scheduler = None
        self._loaded_source = None
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class SdxlLoraTrainer(ZImageLoraTrainer):
    """Generic SDXL-UNet LoRA trainer.

    Reuses :class:`ZImageLoraTrainer`'s backend-agnostic staged orchestration
    (prepare → load → cache → train → checkpoint → save) with the SDXL U-Net
    backend. Trained from a still-image dataset; the output is an ``sdxl`` family
    LoRA the SDXL adapter loads at inference. This is the shared foundation epic
    1929 extends for Kolors (subclass + swap the pipeline class + prompt encoder).
    """

    kernel_id = "sdxl_lora"

    def _create_backend(self) -> _SdxlLoraBackend:
        return _SdxlLoraBackend()


# --------------------------------------------------------------------------- #
# Kolors LoRA backend (epic 1929) — the SDXL-UNet trainer with the Kolors pipeline
# --------------------------------------------------------------------------- #


class _KolorsLoraBackend(_SdxlLoraBackend):
    """Kolors LoRA training backend — the SDXL-UNet trainer with the Kolors pipeline.

    Kolors (Kwai-Kolors) is an SDXL-architecture U-Net with a **ChatGLM3-6B** text
    encoder + SDXL VAE and the same epsilon/v-prediction objective, so it reuses
    every part of :class:`_SdxlLoraBackend` — the U-Net LoRA injection via
    ``build_peft_network_config`` (so LoKr is inherited for free, epic 2193), the
    DDPM loop with the SDXL ``added_cond_kwargs``, and the save path — and only
    swaps the two documented seams: the pipeline class and the prompt encoder
    (``KolorsPipeline.encode_prompt`` — ChatGLM3, ``max_sequence_length=256``, no
    second CLIP ``prompt_2``). After caching the prompt embeddings it releases the
    ~6B-param ChatGLM3 encoder when no live sampling will re-encode prompts,
    keeping the Mac memory envelope to U-Net + LoRA + VAE.
    """

    kernel_id = "kolors_lora"
    pipeline_class_name = "KolorsPipeline"
    # Kolors-diffusers ships fp16-variant weights only (like the SDXL base).
    load_variant = "fp16"

    def _encode_prompt(self, pipe: Any, caption: str, device: str) -> tuple[Any, Any]:
        # Kolors uses a single ChatGLM3 encoder (no SDXL ``prompt_2``) and a GLM
        # sequence length of 256; the 4-tuple return order matches SDXL.
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=caption,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=256,
        )
        return prompt_embeds, pooled_prompt_embeds

    def prepare_dataset(
        self,
        *,
        items: list[dict[str, Any]],
        config: TrainingRunConfig,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        result = super().prepare_dataset(
            items=items, config=config, progress=progress, cancel_requested=cancel_requested
        )
        # ChatGLM3-6B (~12 GB fp16) is only used to encode prompts. The embeddings
        # are now cached and the training loop never touches the encoder, so unless
        # live sampling will re-encode prompts mid-run, release it to keep the Mac
        # memory envelope to U-Net + LoRA + VAE (epic 1929).
        if not config.sample_every:
            self._release_text_encoder()
        return result

    def _release_text_encoder(self) -> None:
        pipe = self._pipeline
        if pipe is None:
            return
        released: list[str] = []
        for attr in ("text_encoder", "text_encoder_2"):
            if getattr(pipe, attr, None) is not None:
                setattr(pipe, attr, None)
                released.append(attr)
        if released:
            empty_torch_cache(self._torch)
            emit_worker_event(
                "training_text_encoder_released",
                kernel=self.kernel_id,
                released=",".join(released),
            )


class KolorsLoraTrainer(SdxlLoraTrainer):
    """Kolors LoRA trainer (epic 1929).

    Thin extension of the generic SDXL-UNet trainer: the same staged orchestration
    and SDXL training loop with the Kolors pipeline + ChatGLM3 prompt encoder via
    :class:`_KolorsLoraBackend`. Output registers as a ``kolors`` family LoRA the
    Kolors image adapter loads at generation time; LoKr support is inherited from
    the shared SDXL backend (epic 2193, sc-2217).
    """

    kernel_id = "kolors_lora"

    def _create_backend(self) -> _KolorsLoraBackend:
        return _KolorsLoraBackend()


# --------------------------------------------------------------------------- #
# Real torch/diffusers/peft backend for Wan2.2 video
# --------------------------------------------------------------------------- #


class _WanLoraBackend:
    """Real Wan2.2 video LoRA training backend (torch/diffusers/peft).

    Loads the diffusers ``WanPipeline`` components, attaches a PEFT LoRA to the
    transformer, caches per-item video latents (Wan-VAE) + umT5 prompt embeddings,
    and runs a flow-matching velocity loop. Trained from a still-image dataset
    (each item encodes to a single latent frame, T=1) — the same approach the
    shipped LTX video LoRA uses — so it reuses the shared image dataset path. The
    latent cache keeps the Wan-VAE 5D shape ``(B, C, T, H, W)``, so a future
    clip dataset can supply ``T > 1`` frames without changing the training loop.
    The 14B MoE trainer (sc-1953) extends this for the two-expert case.

    Wan specifics vs the Z-Image backend (validated in spike sc-1950):
    - ``WanPipeline`` / ``WanTransformer3DModel`` / ``AutoencoderKLWan``; no
      de-distill adapter (Wan2.2-TI2V-5B is not step-distilled).
    - **MPS runs in fp32.** Wan's ``patch_embedding`` (a Conv3d) and the 3D causal
      VAE have no bf16 Metal kernel, so bf16 raises on Apple Silicon; CUDA keeps
      bf16 (or the requested mixed precision).
    - Wan-VAE latent normalization uses per-channel ``latents_mean`` /
      ``latents_std`` (not a single ``scaling_factor`` / ``shift_factor``).
    - The flow-matching target is ``noise - latents``: WanPipeline integrates the
      raw transformer output as the velocity ``d/dt[(1-t)·x0 + t·noise] = noise - x0``
      and does NOT negate it first (unlike ZImagePipeline — see
      ``flow_matching_velocity_target``), so the sign is the opposite of the
      Z-Image backend's.
    - Output registers as a ``wan-video`` family LoRA; the diffusers transformer
      LoRA keys (``transformer.blocks.N.attn1/attn2.*``) are what the inference
      loader (sc-1955) keys 5B-vs-14B gating on.
    """

    kernel_id = "wan_lora"

    def __init__(self) -> None:
        self._torch: Any | None = None
        self._device: str | None = None
        self._dtype: Any | None = None
        self._pipeline: Any | None = None
        self._transformer: Any | None = None
        self._vae: Any | None = None
        self._optimizer: Any | None = None
        self._lr_scheduler: Any | None = None
        self._generator: Any | None = None
        self._loaded_source: str | None = None
        self._latents: list[Any] = []
        self._embeds: list[Any] = []
        self._latents_mean: Any | None = None
        self._latents_std: Any | None = None
        self._diagnosed_forward = False

    def loaded_models(self) -> list[str]:
        return [self._loaded_source] if self._loaded_source else []

    def load(
        self,
        *,
        settings: WorkerSettings,
        plan: dict[str, Any],
        config: TrainingRunConfig,
        progress: ProgressCallback,
    ) -> None:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        peft = importlib.import_module("peft")
        self._torch = torch

        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, config.mixed_precision)
        # Apple Silicon: Wan's Conv3d patch_embedding + 3D VAE have no bf16/fp16
        # Metal kernel, so force fp32 on MPS regardless of the requested precision
        # (sc-1950). CUDA keeps the selected dtype.
        if str(device) == "mps":
            dtype = torch.float32
        source = resolve_pretrained_source(plan.get("target") or {})

        pipeline_class = getattr(diffusers, "WanPipeline", None)
        if pipeline_class is None:
            raise TrainingKernelError(
                "The installed diffusers build does not expose WanPipeline; "
                "install a diffusers build with Wan2.2 support."
            )

        emit_worker_event(
            "training_pipeline_load_start",
            kernel=self.kernel_id,
            source=source,
            device=device,
            dtype=str(dtype),
        )
        progress("loading_model", "loading_model", 0.12, "Loading Wan2.2 base model files.")
        pipe = pipeline_class.from_pretrained(source, torch_dtype=dtype)
        pipe.to(device)

        transformer = pipe.transformer
        transformer.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None:
            text_encoder.requires_grad_(False)

        progress("loading_model", "loading_model", 0.16, "Attaching LoRA adapter to the transformer.")
        self._network_type = config.network_type
        # Stashed for the save path: everything write_lokr_adapter needs to stamp
        # into the file so inference can rebuild the matching LoKrConfig (epic 2193).
        self._lokr_save_kwargs = {
            "rank": config.rank,
            "alpha": config.alpha,
            "decompose_factor": config.decompose_factor,
            "target_modules": config.lora_target_modules,
        }
        lora_config = build_peft_network_config(peft, config)
        transformer.add_adapter(lora_config)
        self._activate_lora_adapter(transformer)
        if config.gradient_checkpointing:
            # Frozen base + LoRA: force inputs to require grad so reentrant
            # checkpointing does not drop gradients to the adapter (its inputs
            # otherwise don't require grad). Mirrors the Z-Image backend.
            if hasattr(transformer, "enable_input_require_grads"):
                try:
                    transformer.enable_input_require_grads()
                except Exception:
                    pass
            if hasattr(transformer, "enable_gradient_checkpointing"):
                transformer.enable_gradient_checkpointing()
            elif hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable()
            else:
                emit_worker_event(
                    "training_gradient_checkpointing_unavailable",
                    kernel=self.kernel_id,
                    transformer=type(transformer).__name__,
                )
        transformer.train()
        trainable = [param for param in transformer.parameters() if param.requires_grad]
        if not trainable:
            raise TrainingKernelError(
                "LoRA adapter attached no trainable parameters; the configured "
                "target modules did not match any transformer layers. Adjust "
                "advanced.loraTargetModules for this base model."
            )

        self._optimizer = build_optimizer(
            config.optimizer, trainable, config.learning_rate, config.weight_decay
        )
        self._optimizer.zero_grad()
        total_updates, warmup_updates = lr_schedule_updates(
            config.steps, config.gradient_accumulation, config.lr_warmup_steps
        )
        self._lr_scheduler = build_lr_scheduler(
            torch,
            self._optimizer,
            config.lr_scheduler,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
        )
        self._latents_mean, self._latents_std = self._vae_normalization(torch, pipe.vae)
        self._pipeline = pipe
        self._transformer = transformer
        self._vae = pipe.vae
        self._device = device
        self._dtype = dtype
        self._loaded_source = source
        generator_device = device if str(device).startswith("cuda") else "cpu"
        self._generator = torch.Generator(generator_device).manual_seed(int(config.seed))
        lora_a_norm, lora_b_norm = self._lora_param_norms()
        emit_worker_event(
            "training_pipeline_load_complete",
            kernel=self.kernel_id,
            source=source,
            trainableTensors=len(trainable),
            loraANorm=lora_a_norm,
            loraBNorm=lora_b_norm,
            lrScheduler=config.lr_scheduler,
            lrWarmupSteps=config.lr_warmup_steps,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

    def _vae_normalization(self, torch: Any, vae: Any) -> tuple[Any | None, Any | None]:
        """Build the Wan-VAE per-channel ``(mean, 1/std)`` latent normalizers as
        ``(1, C, 1, 1, 1)`` tensors, or ``(None, None)`` if the VAE config lacks
        them (then raw latents are used). WanPipeline normalizes encoded latents
        as ``(latent - mean) / std``."""

        cfg = getattr(vae, "config", None)
        mean = getattr(cfg, "latents_mean", None)
        std = getattr(cfg, "latents_std", None)
        if not mean or not std:
            return None, None
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, len(mean), 1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, len(std), 1, 1, 1)
        return mean_t, std_t

    def prepare_dataset(
        self,
        *,
        items: list[dict[str, Any]],
        config: TrainingRunConfig,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        torch = self._torch
        pipe = self._pipeline
        vae = self._vae
        resolution = bucket_resolution(config.resolution)
        count = len(items)
        self._latents = []
        self._embeds = []
        # Encode latents with the VAE in fp32 to avoid NaNs in the 3D causal VAE
        # (matches the SDXL backend's caching upcast); cached latents are cast to
        # the training dtype per step. On MPS the whole stack is already fp32.
        vae_dtype = next(vae.parameters()).dtype
        mean = self._latents_mean
        std = self._latents_std
        with torch.no_grad():
            if vae_dtype != torch.float32:
                vae.to(torch.float32)
            for index, item in enumerate(items):
                if cancel_requested():
                    raise InterruptedError("LoRA training canceled by user.")
                image = _load_training_image(item["imagePath"], resolution)
                pixel = _image_to_tensor(torch, image, torch.float32, self._device)
                # Wan-VAE expects a 5D video tensor (B, C, T, H, W); a still image
                # is a single frame (T=1).
                pixel = pixel.unsqueeze(2)
                latent = vae.encode(pixel).latent_dist.sample(generator=self._generator)
                if mean is not None and std is not None:
                    latent = (latent - mean.to(latent.device)) / std.to(latent.device)
                self._latents.append(latent.detach().to(device="cpu", dtype=torch.float32))

                prompt_embeds, _ = pipe.encode_prompt(
                    prompt=str(item.get("caption") or ""),
                    do_classifier_free_guidance=False,
                    num_videos_per_prompt=1,
                    device=self._device,
                )
                embed = prompt_embeds[0] if isinstance(prompt_embeds, (list, tuple)) else prompt_embeds
                self._embeds.append(embed.detach().to(device="cpu", dtype=torch.float32))

                if (index + 1) % 4 == 0 or index + 1 == count:
                    progress(
                        "running",
                        "caching_latents",
                        _scaled(_CACHE_PROGRESS_START, _CACHE_PROGRESS_END, index + 1, count),
                        f"Encoded {index + 1} of {count} dataset item(s).",
                    )
            if vae_dtype != torch.float32:
                vae.to(vae_dtype)
        return {"itemCount": count, "resolution": resolution}

    def train_step(self, *, step: int, total_steps: int, config: TrainingRunConfig) -> float:
        torch = self._torch
        transformer = self._transformer
        device = self._device
        dtype = self._dtype
        index = (step - 1) % len(self._latents)
        latents = self._latents[index].to(device=device, dtype=dtype)
        embeds = self._embeds[index].to(device=device, dtype=dtype)

        # Flow matching: interpolate clean latent (t=0) toward noise (t=1). Wan's
        # WanPipeline integrates the raw transformer output as the velocity
        # ``noise - latents`` (no negation), so that is the regression target. The
        # timestep is the noise level scaled to [0, 1000] (t=1 -> 1000 = full noise).
        noise = seeded_sample(
            torch, torch.randn, latents.shape, generator=self._generator, device=device, dtype=latents.dtype
        )
        t = sample_training_timestep(
            torch,
            generator=self._generator,
            device=device,
            dtype=latents.dtype,
            timestep_type=config.timestep_type,
            timestep_bias=config.timestep_bias,
        )
        t_broadcast = t.view(-1, *([1] * (latents.dim() - 1)))
        noisy = (1.0 - t_broadcast) * latents + t_broadcast * noise
        target = noise - latents
        timestep = t * 1000.0

        prediction = transformer(
            hidden_states=noisy,
            timestep=timestep,
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]
        if not self._diagnosed_forward:
            emit_worker_event(
                "training_forward_shapes",
                kernel=self.kernel_id,
                latent=list(latents.shape),
                prediction=list(prediction.shape),
                target=list(target.shape),
            )
            self._diagnosed_forward = True

        loss = training_loss(torch, prediction, target, config.loss_type)
        accum = max(1, config.gradient_accumulation)
        (loss / accum).backward()
        if step % accum == 0 or step == total_steps:
            self._optimizer.step()
            apply_weight_noise(torch, self._optimizer, config.weight_noise_sigma)
            self._optimizer.zero_grad()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
        return float(loss.detach().to("cpu"))

    def save_checkpoint(self, *, step: int, output_dir: str, file_name: str) -> str | None:
        stem = Path(file_name).stem or "lora"
        checkpoint_name = f"{stem}-step{step:06d}.safetensors"
        return self._save_lora(output_dir=output_dir, file_name=checkpoint_name)

    def generate_samples(
        self,
        *,
        step: int,
        prompts: list[str],
        output_dir: str,
        file_name: str,
        plan: dict[str, Any],
        config: TrainingRunConfig,
    ) -> list[dict[str, Any]]:
        # In-training video sample rendering is intentionally not implemented for
        # the first cut (Wan video generation per step is expensive); presets set
        # sampleEvery=0 so the orchestration never calls this. Tracked as a
        # follow-up alongside the video-clip dataset work.
        return []

    def save_final(self, *, output_dir: str, file_name: str) -> str:
        return self._save_lora(output_dir=output_dir, file_name=file_name)

    def _save_lora(self, *, output_dir: str, file_name: str) -> str:
        from peft.utils import get_peft_model_state_dict

        os.makedirs(output_dir, exist_ok=True)
        lora_state_dict = get_peft_model_state_dict(self._transformer)
        if getattr(self, "_network_type", "lora") == "lokr":
            # LoKr keys (lokr_w1/lokr_w2) are not save_lora_weights-compatible;
            # write raw with routing metadata (epic 2193).
            write_lokr_adapter(
                lora_state_dict,
                output_dir,
                file_name,
                **getattr(self, "_lokr_save_kwargs", {}),
            )
        else:
            type(self._pipeline).save_lora_weights(
                output_dir,
                transformer_lora_layers=lora_state_dict,
                weight_name=file_name,
                safe_serialization=True,
            )
        lora_a_norm, lora_b_norm = self._lora_param_norms()
        emit_worker_event(
            "training_lora_weight_norm",
            kernel=self.kernel_id,
            fileName=file_name,
            tensors=len(lora_state_dict),
            loraANorm=lora_a_norm,
            loraBNorm=lora_b_norm,
        )
        return os.path.join(output_dir, file_name)

    def _activate_lora_adapter(self, transformer: Any) -> None:
        for method_name in ("set_adapter", "enable_adapters"):
            method = getattr(transformer, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "set_adapter":
                    method("default")
                else:
                    method()
                return
            except Exception as exc:
                emit_worker_event(
                    "training_lora_adapter_activation_failed",
                    kernel=self.kernel_id,
                    method=method_name,
                    error=str(exc),
                )

    def _lora_param_norms(self) -> tuple[float, float]:
        transformer = self._transformer
        if transformer is None or not hasattr(transformer, "named_parameters"):
            return 0.0, 0.0
        a_sq = 0.0
        b_sq = 0.0
        for name, param in transformer.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            try:
                value = float(param.detach().float().pow(2).sum().to("cpu"))
            except Exception:
                continue
            if "lora_B" in name or "lora_b" in name:
                b_sq += value
            elif "lora_A" in name or "lora_a" in name:
                a_sq += value
        return round(a_sq**0.5, 6), round(b_sq**0.5, 6)

    def cleanup(self) -> None:
        torch = self._torch
        self._latents = []
        self._embeds = []
        self._optimizer = None
        self._transformer = None
        self._vae = None
        self._pipeline = None
        self._loaded_source = None
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class WanLoraTrainer(ZImageLoraTrainer):
    """Wan2.2-TI2V-5B video LoRA trainer (torch/diffusers).

    Reuses :class:`ZImageLoraTrainer`'s backend-agnostic staged orchestration
    (prepare → load → cache → train → checkpoint → save) with the torch Wan
    backend. Trained from a still-image dataset (single latent frame); the output
    is a ``wan-video`` family LoRA the Wan video adapter loads at inference. The
    14B MoE trainer (sc-1953) extends this for the two-expert (high/low-noise)
    case.
    """

    kernel_id = "wan_lora"

    def _create_backend(self) -> _WanLoraBackend:
        return _WanLoraBackend()


# --------------------------------------------------------------------------- #
# Real torch/diffusers/peft backend for Wan2.2 A14B (MoE dual-expert) video
# --------------------------------------------------------------------------- #


class _WanMoeLoraBackend(_WanLoraBackend):
    """Wan2.2 A14B MoE dual-expert video LoRA backend.

    Extends the dense 5B Wan backend to the A14B two-expert architecture: a
    high-noise expert (``transformer`` — early/large-timestep denoising) and a
    low-noise expert (``transformer_2`` — late/small-timestep). It trains a
    SEPARATE LoRA on each expert, alternating per training step and sampling each
    expert's timestep WITHIN its own noise band, split at the pipeline's
    ``boundary_ratio`` (0.875 for A14B). Two per-expert safetensors are saved
    (``<name>.high_noise`` / ``<name>.low_noise``) which the inference loader
    applies to the matching expert.

    Expert loading is pluggable:
    - **bf16** (default, CUDA production): both experts come from
      ``WanPipeline.from_pretrained(repo)`` (``transformer`` + ``transformer_2``).
    - **Q8_0 GGUF** (memory-bound hosts incl. Apple-Silicon validation): each
      expert loads via ``WanTransformer3DModel.from_single_file(..., GGUFQuantizationConfig)``
      and injects as ``transformer`` / ``transformer_2``. The spike (sc-1950)
      confirmed a LoRA trains on a GGUF-quantized base. Selected via
      ``advanced.baseQuantization = {"format": "gguf", "repo": ..., "highNoiseFile": ..., "lowNoiseFile": ...}``.

    A14B bf16 (~56GB of transformers + umT5 + VAE) is GPU-only; the GGUF path
    (~28GB for both experts) is what fits a 128GB Mac for desk validation.
    """

    kernel_id = "wan_moe_lora"

    def __init__(self) -> None:
        super().__init__()
        self._hi: Any | None = None
        self._lo: Any | None = None
        self._hi_opt: Any | None = None
        self._lo_opt: Any | None = None
        self._hi_sched: Any | None = None
        self._lo_sched: Any | None = None
        self._boundary: float = 0.875
        self._hi_micro = 0
        self._lo_micro = 0

    def load(
        self,
        *,
        settings: WorkerSettings,
        plan: dict[str, Any],
        config: TrainingRunConfig,
        progress: ProgressCallback,
    ) -> None:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        peft = importlib.import_module("peft")
        self._torch = torch

        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, config.mixed_precision)
        if str(device) == "mps":
            dtype = torch.float32
        source = resolve_pretrained_source(plan.get("target") or {})

        emit_worker_event(
            "training_pipeline_load_start",
            kernel=self.kernel_id,
            source=source,
            device=device,
            dtype=str(dtype),
        )
        progress("loading_model", "loading_model", 0.12, "Loading Wan2.2 A14B experts.")
        pipe, hi, lo = self._load_experts(diffusers, source, dtype, device, config, progress)
        if lo is None:
            raise TrainingKernelError(
                "The Wan MoE trainer requires a two-expert (A14B) model, but the "
                "loaded pipeline has no transformer_2 (low-noise expert)."
            )
        self._boundary = self._resolve_boundary(pipe)

        hi.requires_grad_(False)
        lo.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None:
            text_encoder.requires_grad_(False)

        progress("loading_model", "loading_model", 0.16, "Attaching LoRA adapters to both experts.")
        self._hi_opt, self._hi_sched = self._attach_expert_lora(peft, hi, config, torch)
        self._lo_opt, self._lo_sched = self._attach_expert_lora(peft, lo, config, torch)

        self._latents_mean, self._latents_std = self._vae_normalization(torch, pipe.vae)
        self._pipeline = pipe
        self._hi, self._lo = hi, lo
        # Parent helpers default to self._transformer; point it at the high-noise
        # expert so any inherited diagnostic still works, but saving is per-expert.
        self._transformer = hi
        self._vae = pipe.vae
        self._device = device
        self._dtype = dtype
        self._loaded_source = source
        generator_device = device if str(device).startswith("cuda") else "cpu"
        self._generator = torch.Generator(generator_device).manual_seed(int(config.seed))
        emit_worker_event(
            "training_pipeline_load_complete",
            kernel=self.kernel_id,
            source=source,
            boundaryRatio=self._boundary,
            quantized=self._quant_spec(config) is not None,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

    def _quant_spec(self, config: TrainingRunConfig) -> dict[str, Any] | None:
        """Parse ``advanced.baseQuantization`` into a GGUF expert spec, or None for
        the default bf16 path."""
        spec = (config.advanced or {}).get("baseQuantization")
        if not isinstance(spec, dict):
            return None
        if str(spec.get("format") or "").strip().lower() != "gguf":
            return None
        repo = spec.get("repo")
        hi = spec.get("highNoiseFile")
        lo = spec.get("lowNoiseFile")
        if not (repo and hi and lo):
            return None
        return {"repo": str(repo), "highNoiseFile": str(hi), "lowNoiseFile": str(lo)}

    def _load_experts(
        self, diffusers: Any, source: str, dtype: Any, device: str,
        config: TrainingRunConfig, progress: ProgressCallback,
    ) -> tuple[Any, Any, Any]:
        pipeline_class = getattr(diffusers, "WanPipeline", None)
        if pipeline_class is None:
            raise TrainingKernelError(
                "The installed diffusers build does not expose WanPipeline; "
                "install a diffusers build with Wan2.2 support."
            )
        quant = self._quant_spec(config)
        if quant is not None:
            transformer_class = getattr(diffusers, "WanTransformer3DModel", None)
            gguf_config = getattr(diffusers, "GGUFQuantizationConfig", None)
            if transformer_class is None or gguf_config is None:
                raise TrainingKernelError(
                    "GGUF-base Wan training requires diffusers WanTransformer3DModel "
                    "+ GGUFQuantizationConfig (and the gguf package)."
                )
            from huggingface_hub import hf_hub_download

            def _resolve(file_ref: str) -> str:
                return file_ref if os.path.exists(file_ref) else hf_hub_download(quant["repo"], file_ref)

            progress("loading_model", "loading_model", 0.13, "Loading high-noise expert (GGUF).")
            hi = transformer_class.from_single_file(
                _resolve(quant["highNoiseFile"]),
                quantization_config=gguf_config(compute_dtype=dtype),
                config=source, subfolder="transformer", torch_dtype=dtype,
            )
            progress("loading_model", "loading_model", 0.15, "Loading low-noise expert (GGUF).")
            lo = transformer_class.from_single_file(
                _resolve(quant["lowNoiseFile"]),
                quantization_config=gguf_config(compute_dtype=dtype),
                config=source, subfolder="transformer", torch_dtype=dtype,
            )
            pipe = pipeline_class.from_pretrained(
                source, transformer=hi, transformer_2=lo, torch_dtype=dtype
            )
        else:
            pipe = pipeline_class.from_pretrained(source, torch_dtype=dtype)
            hi = pipe.transformer
            lo = getattr(pipe, "transformer_2", None)
        pipe.to(device)
        return pipe, hi, lo

    def _resolve_boundary(self, pipe: Any) -> float:
        cfg = getattr(pipe, "config", None)
        value = None
        if cfg is not None:
            try:
                value = cfg.get("boundary_ratio") if hasattr(cfg, "get") else getattr(cfg, "boundary_ratio", None)
            except Exception:
                value = None
        if value is None:
            value = getattr(pipe, "boundary_ratio", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.875
        # A null/zero boundary_ratio (dense models) is meaningless for MoE.
        return value if 0.0 < value < 1.0 else 0.875

    def _attach_expert_lora(self, peft: Any, expert: Any, config: TrainingRunConfig, torch: Any) -> tuple[Any, Any]:
        lora_config = peft.LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            init_lora_weights="gaussian",
            target_modules=list(config.lora_target_modules)
            if isinstance(config.lora_target_modules, (list, tuple))
            else config.lora_target_modules,
        )
        expert.add_adapter(lora_config)
        self._activate_lora_adapter(expert)
        if config.gradient_checkpointing:
            if hasattr(expert, "enable_input_require_grads"):
                try:
                    expert.enable_input_require_grads()
                except Exception:
                    pass
            if hasattr(expert, "enable_gradient_checkpointing"):
                expert.enable_gradient_checkpointing()
            elif hasattr(expert, "gradient_checkpointing_enable"):
                expert.gradient_checkpointing_enable()
        expert.train()
        trainable = [param for param in expert.parameters() if param.requires_grad]
        if not trainable:
            raise TrainingKernelError(
                "LoRA adapter attached no trainable parameters on a Wan expert; the "
                "configured target modules matched no layers. Adjust advanced.loraTargetModules."
            )
        optimizer = build_optimizer(config.optimizer, trainable, config.learning_rate, config.weight_decay)
        optimizer.zero_grad()
        # Each expert trains ~half the micro-steps (alternating), so its scheduler
        # decays over half the updates. constant + no warmup yields None (fixed LR).
        total_updates, warmup_updates = lr_schedule_updates(
            config.steps, config.gradient_accumulation, config.lr_warmup_steps
        )
        scheduler = build_lr_scheduler(
            torch, optimizer, config.lr_scheduler,
            total_updates=max(1, total_updates // 2),
            warmup_updates=warmup_updates // 2,
        )
        return optimizer, scheduler

    def prepare_dataset(
        self,
        *,
        items: list[dict[str, Any]],
        config: TrainingRunConfig,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        prepared = super().prepare_dataset(
            items=items, config=config, progress=progress, cancel_requested=cancel_requested
        )
        # Free the umT5 text encoder after caching embeddings: training never needs
        # it again (in-training samples are off), and it is ~11-22GB on the
        # memory-bound A14B path.
        pipe = self._pipeline
        if pipe is not None and getattr(pipe, "text_encoder", None) is not None:
            try:
                pipe.text_encoder = None
                if self._torch is not None and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:
                pass
        return prepared

    def train_step(self, *, step: int, total_steps: int, config: TrainingRunConfig) -> float:
        torch = self._torch
        device = self._device
        dtype = self._dtype
        # Alternate experts so both get balanced updates; each samples a timestep
        # only within its own noise band (split at boundary_ratio).
        high = step % 2 == 1
        if high:
            expert, optimizer, scheduler = self._hi, self._hi_opt, self._hi_sched
            band_lo, band_hi = self._boundary, 1.0
        else:
            expert, optimizer, scheduler = self._lo, self._lo_opt, self._lo_sched
            band_lo, band_hi = 0.0, self._boundary

        index = (step - 1) % len(self._latents)
        latents = self._latents[index].to(device=device, dtype=dtype)
        embeds = self._embeds[index].to(device=device, dtype=dtype)
        noise = seeded_sample(
            torch, torch.randn, latents.shape, generator=self._generator, device=device, dtype=latents.dtype
        )
        t_unit = sample_training_timestep(
            torch, generator=self._generator, device=device, dtype=latents.dtype,
            timestep_type=config.timestep_type, timestep_bias=config.timestep_bias,
        )
        t = band_lo + t_unit * (band_hi - band_lo)
        t_broadcast = t.view(-1, *([1] * (latents.dim() - 1)))
        noisy = (1.0 - t_broadcast) * latents + t_broadcast * noise
        target = noise - latents
        timestep = t * 1000.0

        prediction = expert(
            hidden_states=noisy, timestep=timestep, encoder_hidden_states=embeds, return_dict=False
        )[0]
        if not self._diagnosed_forward:
            emit_worker_event(
                "training_forward_shapes",
                kernel=self.kernel_id,
                expert="high_noise" if high else "low_noise",
                boundaryRatio=self._boundary,
                latent=list(latents.shape),
                prediction=list(prediction.shape),
            )
            self._diagnosed_forward = True

        loss = training_loss(torch, prediction, target, config.loss_type)
        accum = max(1, config.gradient_accumulation)
        (loss / accum).backward()
        if high:
            self._hi_micro += 1
            micro = self._hi_micro
        else:
            self._lo_micro += 1
            micro = self._lo_micro
        if micro % accum == 0 or step >= total_steps - 1:
            optimizer.step()
            apply_weight_noise(torch, optimizer, config.weight_noise_sigma)
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        return float(loss.detach().to("cpu"))

    def save_checkpoint(self, *, step: int, output_dir: str, file_name: str) -> str | None:
        stem = Path(file_name).stem or "lora"
        ext = Path(file_name).suffix or ".safetensors"
        return self._save_both(output_dir=output_dir, stem=f"{stem}-step{step:06d}", ext=ext)

    def save_final(self, *, output_dir: str, file_name: str) -> str:
        stem = Path(file_name).stem or "lora"
        ext = Path(file_name).suffix or ".safetensors"
        return self._save_both(output_dir=output_dir, stem=stem, ext=ext)

    def _save_both(self, *, output_dir: str, stem: str, ext: str) -> str:
        from peft.utils import get_peft_model_state_dict

        os.makedirs(output_dir, exist_ok=True)
        primary: str | None = None
        for expert, suffix in ((self._hi, "high_noise"), (self._lo, "low_noise")):
            name = f"{stem}.{suffix}{ext}"
            lora_state_dict = get_peft_model_state_dict(expert)
            type(self._pipeline).save_lora_weights(
                output_dir,
                transformer_lora_layers=lora_state_dict,
                weight_name=name,
                safe_serialization=True,
            )
            a_norm, b_norm = self._expert_lora_norms(expert)
            emit_worker_event(
                "training_lora_weight_norm",
                kernel=self.kernel_id,
                fileName=name,
                expert=suffix,
                tensors=len(lora_state_dict),
                loraANorm=a_norm,
                loraBNorm=b_norm,
            )
            path = os.path.join(output_dir, name)
            if primary is None:
                primary = path
        # Both files are written; the high-noise path is reported as the primary
        # output (the loader discovers the low-noise sibling by the naming pair).
        return primary or os.path.join(output_dir, f"{stem}{ext}")

    def _expert_lora_norms(self, expert: Any) -> tuple[float, float]:
        if expert is None or not hasattr(expert, "named_parameters"):
            return 0.0, 0.0
        a_sq = 0.0
        b_sq = 0.0
        for name, param in expert.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            try:
                value = float(param.detach().float().pow(2).sum().to("cpu"))
            except Exception:
                continue
            if "lora_B" in name or "lora_b" in name:
                b_sq += value
            elif "lora_A" in name or "lora_a" in name:
                a_sq += value
        return round(a_sq**0.5, 6), round(b_sq**0.5, 6)

    def cleanup(self) -> None:
        self._hi = None
        self._lo = None
        self._hi_opt = None
        self._lo_opt = None
        super().cleanup()


class WanMoeLoraTrainer(ZImageLoraTrainer):
    """Wan2.2 A14B MoE dual-expert video LoRA trainer (torch/diffusers).

    Reuses :class:`ZImageLoraTrainer`'s staged orchestration with the two-expert
    Wan MoE backend. Trains a separate LoRA on the high-noise and low-noise
    experts (split at the pipeline's ``boundary_ratio``), saving two ``wan-video``
    family safetensors the inference loader applies per expert. Extends the dense
    5B :class:`WanLoraTrainer`'s recipe. GPU-only for the bf16 base; the Q8_0 GGUF
    base path fits memory-bound hosts.
    """

    kernel_id = "wan_moe_lora"

    def _create_backend(self) -> _WanMoeLoraBackend:
        return _WanMoeLoraBackend()


class LensLoraTrainer:
    """Image LoRA trainer for Microsoft Lens, run OUT-OF-PROCESS.

    Lens needs transformers 5.x + diffusers 0.38, incompatible with this (main)
    worker venv's transformers 4.x stack, so the whole training loop runs in the
    dedicated Lens sidecar venv via ``scene_worker/lens_train_runner.py``. (Lens
    *inference* was ported to the native candle backend and its sidecar adapter
    retired in sc-5126; training stays in the venv until the sc-5147 cutover.)
    Per-step IPC across the venv boundary
    would be far too chatty, so unlike :class:`ZImageLoraTrainer` (which stages an
    in-process backend) this driver only writes the spec, launches the subprocess,
    maps its JSONL progress events onto the worker's progress bands, handles
    cancellation, and shapes the result. The output is a ``lens`` family LoRA the
    Lens adapter applies to Lens-Turbo at inference (sc-1587).
    """

    kernel_id = "lens_lora"

    def loaded_models(self) -> list[str]:
        # The sidecar loads and frees the base model per job; nothing resident here.
        return []

    def discard_temp_outputs(self, job_id: str | None = None) -> None:
        """Reap the in-flight training scratch dir only — filesystem-only.

        Called from train's finally and from the force-cancel monitor thread right
        before os._exit, so it must stay filesystem-only (no torch/GPU; the main
        thread may be wedged in a native call). A trainer is created per job and
        the worker runs one job at a time, so a single scratch dir suffices (sc-1719)."""
        work_dir = getattr(self, "_scratch_dir", None)
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._scratch_dir = None

    @staticmethod
    def _lens_python() -> str:
        return os.getenv("SCENEWORKS_LENS_PYTHON", "/opt/lens-venv/bin/python")

    @staticmethod
    def _runner_path() -> Path:
        return Path(__file__).resolve().parent / "lens_train_runner.py"

    def _sidecar_available(self) -> bool:
        return Path(self._lens_python()).exists() and self._runner_path().exists()

    @staticmethod
    def _device_hint(settings: WorkerSettings) -> str:
        """Resolve the device the sidecar should train on via the shared
        ``select_torch_device``: it picks ``mps`` on Apple Silicon, ``cuda``/``cuda:N``
        on NVIDIA, and ``cpu``
        when ``SCENEWORKS_GPU_ID=cpu`` is set. The driver runs in the main worker
        venv (which has torch); if torch is somehow unimportable here we fall back
        to a platform heuristic so the hint is still sane (the sidecar re-resolves
        and fails fast on a real mismatch)."""
        gpu_id = getattr(settings, "gpu_id", None)
        try:
            torch = importlib.import_module("torch")
            return select_torch_device(torch, gpu_id)
        except ImportError:
            token = str(gpu_id or "").strip()
            if token.lower() == "cpu":
                return "cpu"
            if sys.platform == "darwin" and platform.machine() == "arm64":
                return "mps"
            return f"cuda:{token}" if token.isdigit() else "cuda"

    def train(
        self,
        *,
        settings: WorkerSettings,
        plan: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        items = validate_training_plan(plan, require_images=True)
        config = read_run_config(plan)
        output = plan.get("output") or {}
        target = plan.get("target") or {}
        output_dir = str(output.get("outputDir") or "")
        file_name = str(output.get("fileName") or "lora.safetensors")
        if not output_dir:
            raise TrainingKernelError("Training plan output is missing an output directory.")
        if not self._sidecar_available():
            raise TrainingKernelError(
                "Lens LoRA training requires the isolated Lens sidecar venv. Rebuild the worker "
                "image with INCLUDE_LENS=1 (the Docker Compose default), or set SCENEWORKS_LENS_PYTHON "
                f"to a Python interpreter that has the lens stack installed (looked for {self._lens_python()})."
            )
        # The sidecar re-validates, but fail fast here for a plan handed directly
        # to the worker with an unsupported LR scheduler.
        normalize_lr_scheduler(config.lr_scheduler)

        source = resolve_pretrained_source(target)
        os.makedirs(output_dir, exist_ok=True)
        progress("preparing", "preparing", 0.04, "Preparing Lens LoRA training run.")

        work_dir = Path(tempfile.mkdtemp(prefix="lens_train_"))
        self._scratch_dir = work_dir
        progress_path = work_dir / "progress.jsonl"
        progress_path.write_text("", encoding="utf-8")
        result_path = work_dir / "result.json"
        spec = {
            "source": source,
            "device": self._device_hint(settings),
            "dtype": str(config.mixed_precision or "bfloat16"),
            "disableMxfp4": bool((config.advanced or {}).get("disableMxfp4", False)),
            "outputDir": output_dir,
            "fileName": file_name,
            "config": asdict(config),
            "items": [
                {"imagePath": item.get("imagePath"), "caption": item.get("caption") or ""}
                for item in items
            ],
            "samplePrompts": list(config.sample_prompts or []),
            "progressPath": str(progress_path),
            "resultPath": str(result_path),
        }
        spec_path = work_dir / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        total_steps = max(1, config.steps)
        stdout_log = work_dir / "stdout.log"
        cmd = [self._lens_python(), str(self._runner_path()), str(spec_path)]
        emit_worker_event(
            "lens_train_sidecar_start",
            kernel=self.kernel_id,
            source=source,
            steps=total_steps,
            datasetItemCount=len(items),
            sidecar=self._lens_python(),
        )
        progress(
            "loading_model",
            "loading_model",
            0.1,
            f"Loading Lens base model for {target.get('targetId') or self.kernel_id}.",
        )
        try:
            # stdout -> file (avoids any pipe-fill deadlock on a long run); stderr
            # inherits to the worker log. Poll so the job stays cancelable and so we
            # can tail the runner's JSONL progress file between waits.
            with stdout_log.open("w", encoding="utf-8") as out:
                proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=out, stderr=None)
                cursor = 0
                while True:
                    try:
                        proc.wait(timeout=2)
                        self._drain_progress(progress_path, cursor, progress, total_steps)
                        break
                    except subprocess.TimeoutExpired:
                        cursor = self._drain_progress(progress_path, cursor, progress, total_steps)
                        if cancel_requested():
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            raise InterruptedError("LoRA training canceled by user.")
            result = self._read_result(result_path, stdout_log)
            if proc.returncode != 0 or "error" in result:
                error = result.get("error") or f"Lens training sidecar exited with code {proc.returncode}."
                emit_worker_event(
                    "lens_train_sidecar_failed",
                    kernel=self.kernel_id,
                    error=error,
                    returnCode=proc.returncode,
                )
                raise TrainingKernelError(f"Lens LoRA training failed in the sidecar venv: {error}")
            output_path = str(result.get("outputPath") or "")
            if not output_path or not os.path.exists(output_path):
                raise TrainingKernelError(
                    f"Lens training sidecar reported success but produced no adapter at {output_path!r}."
                )
            emit_worker_event(
                "lens_train_sidecar_complete",
                kernel=self.kernel_id,
                outputPath=output_path,
                stepsCompleted=result.get("stepsCompleted"),
            )
            progress("saving", "saving", 0.97, "Saving trained LoRA weights.")
            return self._result_summary(
                plan=plan, config=config, result=result, total_steps=total_steps, items=items
            )
        finally:
            self.discard_temp_outputs()

    @staticmethod
    def _drain_progress(
        path: Path, cursor: int, progress: ProgressCallback, total_steps: int
    ) -> int:
        """Forward any progress lines past ``cursor`` and return the new cursor."""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return cursor
        for line in lines[cursor:]:
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except ValueError:
                continue
            LensLoraTrainer._forward_event(event, progress, total_steps)
        return len(lines)

    @staticmethod
    def _forward_event(event: dict[str, Any], progress: ProgressCallback, total_steps: int) -> None:
        kind = event.get("event")
        if kind == "stage":
            stage = str(event.get("stage") or "running")
            message = str(event.get("message") or "")
            if stage == "caching_latents":
                progress("running", "caching_latents", _CACHE_PROGRESS_START, message)
            elif stage == "training":
                progress("running", "training", _TRAIN_PROGRESS_START, message)
            elif stage == "checkpointing":
                progress("running", "checkpointing", _TRAIN_PROGRESS_START, message)
            elif stage == "saving":
                progress("saving", "saving", 0.95, message)
            else:
                progress("loading_model", "loading_model", 0.12, message)
        elif kind == "cache":
            done = int(event.get("done") or 0)
            total = max(1, int(event.get("total") or 1))
            progress(
                "running",
                "caching_latents",
                _scaled(_CACHE_PROGRESS_START, _CACHE_PROGRESS_END, done, total),
                f"Encoded {done} of {total} dataset item(s).",
            )
        elif kind == "step":
            step = int(event.get("step") or 0)
            loss = event.get("loss")
            progress(
                "running",
                "training",
                _scaled(_TRAIN_PROGRESS_START, _TRAIN_PROGRESS_END, step, total_steps),
                _training_message(step, total_steps, float(loss) if loss is not None else None),
            )
        elif kind == "sample":
            step = int(event.get("step") or 0)
            samples = list(event.get("samples") or [])
            progress(
                "running",
                "rendering",
                _scaled(_TRAIN_PROGRESS_START, _TRAIN_PROGRESS_END, step, total_steps),
                f"Rendered {len(samples)} training sample(s) at step {step}.",
                {"latestTrainingSamples": samples},
            )

    @staticmethod
    def _read_result(result_path: Path, stdout_log: Path) -> dict[str, Any]:
        if result_path.exists():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        try:
            lines = [line for line in stdout_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            lines = []
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                continue
        return {"error": "Lens training sidecar produced no parseable result."}

    def _result_summary(
        self,
        *,
        plan: dict[str, Any],
        config: TrainingRunConfig,
        result: dict[str, Any],
        total_steps: int,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = plan.get("dataset") or {}
        target = plan.get("target") or {}
        output = plan.get("output") or {}
        training_samples = list(result.get("trainingSamples") or [])
        return {
            "mode": "train",
            "kernel": self.kernel_id,
            "loraId": output.get("loraId"),
            "outputDir": output.get("outputDir"),
            "fileName": output.get("fileName"),
            "outputPath": result.get("outputPath"),
            "format": output.get("format") or "safetensors",
            "datasetId": dataset.get("datasetId"),
            "datasetVersion": dataset.get("datasetVersion"),
            "datasetItemCount": len(items),
            "targetId": target.get("targetId"),
            "baseModel": target.get("baseModel"),
            "baseModelSource": result.get("baseModelSource"),
            "steps": total_steps,
            "stepsCompleted": result.get("stepsCompleted") or total_steps,
            "checkpoints": list(result.get("checkpoints") or []),
            "trainingSamples": training_samples,
            "latestTrainingSamples": training_samples[-4:],
            "samplePrompts": config.sample_prompts,
            "sampleSettings": {
                "numInferenceSteps": config.sample_steps,
                "guidanceScale": config.sample_guidance_scale,
                # Lens previews render on the multi-step base model, not Lens-Turbo.
                "sampleSource": "live_adapter_base",
            },
            "rank": config.rank,
            "alpha": config.alpha,
            "learningRate": config.learning_rate,
            "lrScheduler": config.lr_scheduler,
            "lrWarmupSteps": config.lr_warmup_steps,
            "resolution": result.get("resolution") or config.resolution,
            "triggerWords": trigger_words(plan),
            "planVersion": plan.get("planVersion"),
            "completedAt": utc_now(),
        }


# --------------------------------------------------------------------------- #
# Kernel registry
# --------------------------------------------------------------------------- #


_TRAINING_KERNELS: dict[str, Callable[[], Any]] = {
    ZImageLoraTrainer.kernel_id: ZImageLoraTrainer,
    SdxlLoraTrainer.kernel_id: SdxlLoraTrainer,
    KolorsLoraTrainer.kernel_id: KolorsLoraTrainer,
    WanLoraTrainer.kernel_id: WanLoraTrainer,
    WanMoeLoraTrainer.kernel_id: WanMoeLoraTrainer,
    LensLoraTrainer.kernel_id: LensLoraTrainer,
}


def create_training_kernel(kernel_id: Any) -> Any:
    """Return the trainer for a plan's ``target.kernel`` id, or raise for an
    unknown kernel. Mirrors the image/video adapter factories."""

    factory = _TRAINING_KERNELS.get(str(kernel_id or "").strip())
    if factory is None:
        known = ", ".join(sorted(_TRAINING_KERNELS)) or "(none)"
        raise TrainingKernelError(
            f"No training kernel for {kernel_id!r}. Known kernels: {known}."
        )
    return factory()
