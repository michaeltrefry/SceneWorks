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

from dataclasses import dataclass, field
import contextlib
import importlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Protocol

from sceneworks_shared import utc_now

from .adapter_utils import filter_call_kwargs
from .image_adapters import (
    activate_torch_device,
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


def now() -> str:
    return utc_now()


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
        "completedAt": now(),
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
        sample_guidance_scale=_as_float(advanced.get("sampleGuidanceScale"), 0.0),
        sample_prompts=[str(prompt).strip() for prompt in sample_prompts if str(prompt).strip()][:4],
        training_adapter_repo=_as_optional_str(advanced.get("trainingAdapterRepo")),
        training_adapter_version=_as_optional_str(advanced.get("trainingAdapterVersion")),
        advanced=advanced,
    )


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
            "completedAt": now(),
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
    ``prodigyopt`` use the Prodigy optimizer package."""

    torch = importlib.import_module("torch")
    normalized = (name or "").strip().lower().replace("-", "").replace("_", "")
    if normalized in {"prodigy", "prodigyopt"}:
        try:
            prodigy_module = importlib.import_module("prodigyopt")
        except Exception as exc:
            raise TrainingKernelError("The Prodigy optimizer requires the prodigyopt Python package.") from exc
        use_lr = learning_rate if learning_rate >= 0.1 else 1.0
        return prodigy_module.Prodigy(params, lr=use_lr, eps=1e-6, weight_decay=weight_decay)
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
        lora_config = peft.LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            init_lora_weights="gaussian",
            target_modules=list(config.lora_target_modules)
            if isinstance(config.lora_target_modules, (list, tuple))
            else config.lora_target_modules,
        )
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
                            "createdAt": now(),
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
# Native MLX LTX-2.3 LoRA backend (Apple Silicon)
# --------------------------------------------------------------------------- #

LTX_MLX_REPO = "notapalindrome/ltx23-mlx-av-q4"
LTX_MLX_TEXT_ENCODER_REPO = "mlx-community/gemma-3-12b-it-bf16"
# Spatial compression of the LTX VAE: latent edge = pixel edge // 32.
LTX_SPATIAL_SCALE = 32


def require_mlx_runtime() -> None:
    """The LTX kernel runs a native MLX (``mlx.core``) QLoRA loop, which only
    exists on Apple Silicon. Fail with a clear, user-facing message instead of a
    deep import traceback when run elsewhere.

    Gating (sc-1538): the Rust ``ltx_video_lora`` target is marked MLX/Apple-
    Silicon-only so non-Mac clients never offer it. Worker capabilities are
    coarse — any backend-capable GPU worker advertises ``lora_train_execute`` and
    the kernel is resolved at execution — so this is the in-kernel backstop that
    rejects, fast and clearly, an LTX job that still reaches a non-Apple-Silicon
    worker."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise TrainingKernelError(
            "LTX-2.3 LoRA training requires Apple Silicon (macOS arm64); this "
            "worker cannot run the ltx_mlx_lora kernel."
        )
    try:
        importlib.import_module("mlx.core")
        importlib.import_module("mlx_video")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise TrainingKernelError(
            "LTX-2.3 LoRA training requires the optional MLX worker dependencies "
            "(apps/worker/requirements-mlx.txt). " + str(exc)
        ) from exc


def _build_ltx_av_config(
    model_path: Path, config_cls: Any, model_type_cls: Any, rope_type_cls: Any
) -> Any:
    """Build the AudioVideo ``LTXModelConfig`` for the distilled Q4 repo.

    Mirrors ``mlx_video.generate_av``'s inline config: caption projection off →
    ``caption_channels = connector heads * head_dim``; gated attention →
    ``adaln_embedding_coefficient = 9``. Coupled to the installed
    mlx-video-with-audio build; revisit on dependency bumps.
    """
    caption_channels = 3840
    audio_caption_channels = 3840
    caption_proj_first = True
    caption_proj_second = True
    apply_gated = False
    adaln_coeff = 6
    embedded = model_path / "embedded_config.json"
    if embedded.exists():
        try:
            transformer_cfg = json.loads(embedded.read_text()).get("transformer", {})
        except (json.JSONDecodeError, OSError):
            transformer_cfg = {}
        caption_proj_first = transformer_cfg.get("caption_projection_first_linear", True)
        caption_proj_second = transformer_cfg.get("caption_projection_second_linear", True)
        apply_gated = bool(transformer_cfg.get("apply_gated_attention", False))
        adaln_coeff = 9 if apply_gated else 6
        if not caption_proj_first and not caption_proj_second:
            caption_channels = int(transformer_cfg.get("connector_num_attention_heads", 32)) * int(
                transformer_cfg.get("connector_attention_head_dim", 128)
            )
            audio_caption_channels = int(
                transformer_cfg.get("audio_connector_num_attention_heads", 32)
            ) * int(transformer_cfg.get("audio_connector_attention_head_dim", 64))
        else:
            caption_channels = transformer_cfg.get("caption_channels", caption_channels)
            audio_caption_channels = transformer_cfg.get(
                "audio_caption_channels", audio_caption_channels
            )
    return config_cls(
        model_type=model_type_cls.AudioVideo,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=caption_channels,
        caption_projection_first_linear=caption_proj_first,
        caption_projection_second_linear=caption_proj_second,
        adaln_embedding_coefficient=adaln_coeff,
        apply_gated_attention=apply_gated,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_cross_attention_dim=2048,
        audio_caption_channels=audio_caption_channels,
        rope_type=rope_type_cls.SPLIT,
        double_precision_rope=True,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        audio_positional_embedding_max_pos=[20],
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1000,
    )


def load_ltx_transformer(model_path: Path) -> tuple[Any, Any]:
    """Build → selectively quantize → ``load_weights`` the LTX AudioVideo
    transformer exactly as ``mlx_video.generate_av`` does for the split/quantized
    repo. Returns ``(transformer, config)``."""
    mx = importlib.import_module("mlx.core")
    nn = importlib.import_module("mlx.nn")
    generate_av = importlib.import_module("mlx_video.generate_av")
    config_mod = importlib.import_module("mlx_video.models.ltx.config")
    ltx_mod = importlib.import_module("mlx_video.models.ltx.ltx")

    sanitized = generate_av.load_unified_weights(model_path, "transformer.")
    config = _build_ltx_av_config(
        model_path, config_mod.LTXModelConfig, config_mod.LTXModelType, config_mod.LTXRopeType
    )
    transformer = ltx_mod.LTXModel(config)

    manifest_path = model_path / "split_model.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            manifest = {}
        if manifest.get("quantized", False):
            q_bits = int(manifest.get("quantization_bits", 4))
            q_group = int(manifest.get("quantization_group_size", 64))
            quantized_paths = {
                key.rsplit(".", 1)[0] for key in sanitized if key.endswith(".scales")
            }

            def _should_quantize(path: str, module: Any) -> bool:
                return isinstance(module, nn.Linear) and path in quantized_paths

            nn.quantize(transformer, group_size=q_group, bits=q_bits, class_predicate=_should_quantize)

    transformer.load_weights(list(sanitized.items()), strict=False)
    mx.eval(transformer.parameters())
    return transformer, config


def _get_submodule(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_submodule(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _linear_io_dims(base: Any, nn: Any) -> tuple[int, int]:
    """Return (in_features, out_features) for an ``nn.Linear`` or quantized linear."""
    if isinstance(base, nn.QuantizedLinear):
        out_features = base.weight.shape[0]
        in_features = base.weight.shape[1] * (32 // base.bits)
        return in_features, out_features
    weight = base.weight  # (out, in)
    return weight.shape[1], weight.shape[0]


_LORA_LINEAR_CLS: Any = None


def _lora_linear_cls() -> Any:
    """Lazily build the MLX LoRA linear wrapper class (mlx imports are deferred so
    this module stays importable without MLX). Frozen base + trainable rank-r
    A/B, matching the inference loader's math: ``base(x) + (x @ Aᵀ @ Bᵀ) * scale``.
    ``B`` zero-init so the adapter starts as identity."""
    global _LORA_LINEAR_CLS
    if _LORA_LINEAR_CLS is None:
        mx = importlib.import_module("mlx.core")
        nn = importlib.import_module("mlx.nn")

        class _MlxLoRALinear(nn.Module):
            def __init__(self, base: Any, in_features: int, out_features: int, rank: int, alpha: float) -> None:
                super().__init__()
                self.base = base
                self.scale = float(alpha) / float(rank)
                self.lora_a = mx.random.normal((rank, in_features)) * 0.02
                self.lora_b = mx.zeros((out_features, rank))

            def __call__(self, x: Any) -> Any:
                return self.base(x) + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale

        _LORA_LINEAR_CLS = _MlxLoRALinear
    return _LORA_LINEAR_CLS


def inject_video_attention_lora(transformer: Any, config: TrainingRunConfig) -> list[str]:
    """Inject trainable rank-r LoRA into the video self/cross-attention linear
    projections (``config.lora_target_modules`` suffixes under ``attn1``/``attn2``),
    freeze the base, and unfreeze only the LoRA params. Audio and AV-cross modules
    are skipped — they never run in the still-image (``audio=None``) forward.

    Returns the injected real module paths (used verbatim as save keys so the
    inference loader round-trips without remapping)."""
    nn = importlib.import_module("mlx.nn")
    lora_cls = _lora_linear_cls()
    suffixes = set(config.lora_target_modules or DEFAULT_LORA_TARGET_MODULES)

    target_paths = [
        name
        for name, module in transformer.named_modules()
        if isinstance(module, (nn.Linear, nn.QuantizedLinear))
        and name.rsplit(".", 1)[-1] in suffixes
        and (".attn1." in name or ".attn2." in name)
    ]
    if not target_paths:
        raise TrainingKernelError(
            "LTX LoRA injection matched no attention projections; check "
            "advanced.loraTargetModules for this model."
        )

    transformer.freeze()
    for path in target_paths:
        base = _get_submodule(transformer, path)
        in_features, out_features = _linear_io_dims(base, nn)
        wrapper = lora_cls(base, in_features, out_features, config.rank, config.alpha)
        _set_submodule(transformer, path, wrapper)
        _get_submodule(transformer, path).unfreeze(recurse=False, keys=["lora_a", "lora_b"])
    return target_paths


def _build_mlx_optimizer(name: str, learning_rate: Any, weight_decay: float = 0.0) -> Any:
    # ``learning_rate`` is a float (constant) or a schedule callable (built by
    # ``_build_mlx_lr_schedule``); both are valid optimizer inputs and MLX advances
    # a schedule callable from the optimizer's own step counter.
    optim = importlib.import_module("mlx.optimizers")
    normalized = (name or "").strip().lower().replace("-", "").replace("_", "")
    if normalized == "adam":
        # Plain Adam has no decoupled weight decay; the parameter applies to AdamW only.
        return optim.Adam(learning_rate=learning_rate)
    return optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)


def _build_mlx_lr_schedule(
    name: str | None, base_lr: float, *, total_updates: int, warmup_updates: int
) -> Any:
    """Resolve the learning rate handed to the MLX optimizer: a plain float for a
    plain ``constant`` schedule (byte-identical to the pre-scheduler path), or a
    schedule callable that ramps/decays per optimizer update.

    The callable delegates to :func:`lr_decay_multiplier` — the *same* helper the
    torch ``LambdaLR`` uses — so both backends honor identical curves. In
    particular the warmup ramp starts nonzero (``1/(warmup+1)`` of the base LR),
    never wasting the first optimizer update on a 0 LR the way a plain
    ``linear_schedule(0, base, warmup)`` would. MLX advances the schedule from the
    optimizer's own step counter (which increments once per optimizer update), so
    the train loop never steps it manually.

    The callable must return an ``mx.array`` (not a Python float): MLX stores the
    schedule's return value straight into ``optimizer.state["learning_rate"]`` and
    then calls ``.astype(grad.dtype)`` on it inside ``apply_single`` — a Python
    float would raise ``AttributeError`` on the first update. Eager-only: the
    callable reads the step as a Python int, so it must not be traced under
    ``mx.compile``."""

    normalized = normalize_lr_scheduler(name)
    total = max(1, int(total_updates))
    warmup = max(0, min(int(warmup_updates), total - 1))
    if normalized == "constant" and warmup == 0:
        return float(base_lr)

    mx = importlib.import_module("mlx.core")
    base = float(base_lr)

    def schedule(step: Any) -> Any:
        return mx.array(base * lr_decay_multiplier(normalized, int(step), total, warmup))

    return schedule


def ltx_flow_target(clean: Any, noise: Any) -> Any:
    """Rectified-flow velocity target for the RAW LTX transformer output.

    LTX denoises with ``to_denoised(x_t, v, sigma) = x_t - sigma*v`` over the
    schedule ``x_t = (1 - sigma)*x_0 + sigma*noise`` (mlx_video). Solving for the
    velocity that recovers ``x_0`` gives ``v = noise - x_0`` — and the pipeline
    feeds the raw transformer output straight into ``to_denoised`` (no negation,
    unlike diffusers' Z-Image), so that is exactly the regression target."""
    return noise - clean


@contextlib.contextmanager
def _silence_output_fds() -> Any:
    """Redirect OS-level stdout+stderr (fds 1 and 2) to /dev/null for the block.

    The MLX generation pipeline prints chunked-eval / stage progress straight to
    the file descriptors (it logs to stderr), which Python-level
    ``contextlib.redirect_stdout`` does not capture — so mid-training previews
    would otherwise flood the worker log with hundreds of lines per render. A
    failed generation still raises a Python exception (caught by the caller), so
    suppressing the stream text here loses no actionable error signal."""
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


class _LtxMlxLoraBackend:
    """Native MLX LoRA training backend for LTX-2.3 (Apple Silicon).

    Loads the quantized AudioVideo LTX transformer plus the LTX VAE encoder and
    gemma text encoder, then caches a still-image dataset as single-frame latents
    and caption context embeddings. LoRA injection, the flow-matching training
    step, and adapter saving land in sc-1536/sc-1537; this backend covers loading
    and dataset preparation.
    """

    def __init__(self) -> None:
        self._mx: Any | None = None
        self._model_path: Path | None = None
        self._transformer: Any | None = None
        self._config: Any | None = None
        self._vae: Any | None = None
        self._text_encoder: Any | None = None
        self._loaded_source: str | None = None
        self._latents: list[Any] = []
        self._embeds: list[Any] = []
        self._positions: Any | None = None
        self._optimizer: Any | None = None
        self._lora_paths: list[str] = []
        self._accumulated_grads: Any | None = None

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
        require_mlx_runtime()
        mx = importlib.import_module("mlx.core")
        utils = importlib.import_module("mlx_video.utils")
        encoder_mod = importlib.import_module("mlx_video.models.ltx.video_vae.encoder")
        text_mod = importlib.import_module("mlx_video.models.ltx.text_encoder")
        self._mx = mx

        target = plan.get("target") or {}
        advanced = config.advanced if isinstance(config.advanced, dict) else {}
        repo = str(target.get("baseModelRepo") or LTX_MLX_REPO)
        text_repo = str(advanced.get("textEncoderRepo") or LTX_MLX_TEXT_ENCODER_REPO)

        progress("loading_model", "loading_model", 0.1, "Resolving LTX-2.3 (MLX Q4) model files.")
        model_path = Path(utils.get_model_path(repo))
        self._model_path = model_path

        emit_worker_event(
            "training_pipeline_load_start",
            kernel=LtxMlxLoraTrainer.kernel_id,
            source=repo,
            device="mps",
        )
        progress("loading_model", "loading_model", 0.12, "Loading LTX-2.3 transformer (quantized).")
        transformer, ltx_config = load_ltx_transformer(model_path)

        progress("loading_model", "loading_model", 0.15, "Loading LTX VAE encoder.")
        vae = encoder_mod.load_vae_encoder(str(model_path), use_unified=True)
        mx.eval(vae.parameters())

        progress("loading_model", "loading_model", 0.17, "Loading text encoder.")
        text_encoder = text_mod.LTX2TextEncoder()
        text_encoder.load(
            model_path=str(model_path),
            text_encoder_path=str(Path(utils.get_model_path(text_repo))),
            use_unified=True,
        )
        mx.eval(text_encoder.parameters())

        progress("loading_model", "loading_model", 0.18, "Attaching LoRA adapters to the transformer.")
        lora_paths = inject_video_attention_lora(transformer, config)
        # Resolve the LR (constant float, or a schedule callable that decays/ramps
        # per optimizer update — MLX advances it from the optimizer's step count).
        total_updates, warmup_updates = lr_schedule_updates(
            config.steps, config.gradient_accumulation, config.lr_warmup_steps
        )
        learning_rate = _build_mlx_lr_schedule(
            config.lr_scheduler,
            config.learning_rate,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
        )
        self._optimizer = _build_mlx_optimizer(
            config.optimizer, learning_rate, config.weight_decay
        )
        self._accumulated_grads = None

        self._transformer = transformer
        self._config = ltx_config
        self._vae = vae
        self._text_encoder = text_encoder
        self._loaded_source = repo
        self._lora_paths = lora_paths
        emit_worker_event(
            "training_pipeline_load_complete",
            kernel=LtxMlxLoraTrainer.kernel_id,
            source=repo,
            loraModules=len(lora_paths),
            lrScheduler=config.lr_scheduler,
            lrWarmupSteps=config.lr_warmup_steps,
        )

    def prepare_dataset(
        self,
        *,
        items: list[dict[str, Any]],
        config: TrainingRunConfig,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        mx = self._mx
        generate_mod = importlib.import_module("mlx_video.generate")
        encoder_mod = importlib.import_module("mlx_video.models.ltx.video_vae.encoder")
        utils = importlib.import_module("mlx_video.utils")

        resolution = bucket_resolution(config.resolution)
        latent_edge = max(1, resolution // LTX_SPATIAL_SCALE)
        count = len(items)
        self._latents = []
        self._embeds = []
        # Single still frame → one latent frame; positions are identical across
        # items at a fixed resolution, so compute the grid once.
        self._positions = generate_mod.create_position_grid(1, 1, latent_edge, latent_edge)

        for index, item in enumerate(items):
            _check_cancel(cancel_requested)
            image = utils.load_image(item["imagePath"], height=resolution, width=resolution)
            latent = encoder_mod.encode_image(image, self._vae)  # (1, 128, 1, h, w)
            batch, channels = latent.shape[0], latent.shape[1]
            flat = mx.transpose(mx.reshape(latent, (batch, channels, -1)), (0, 2, 1))
            mx.eval(flat)
            self._latents.append(flat)

            video_embeds, _ = self._text_encoder(
                str(item.get("caption") or ""), return_audio_embeddings=False
            )
            mx.eval(video_embeds)
            self._embeds.append(video_embeds)

            if (index + 1) % 4 == 0 or index + 1 == count:
                progress(
                    "running",
                    "caching_latents",
                    _scaled(_CACHE_PROGRESS_START, _CACHE_PROGRESS_END, index + 1, count),
                    f"Encoded {index + 1} of {count} dataset item(s).",
                )
        # The gemma text encoder (~28 GB) is only needed to cache the caption
        # embeds above; the train loop runs the transformer alone. Release it now
        # so it is not resident through training — drops the training-loop peak
        # from ~59 GB to ~27 GB (the whole-run ceiling becomes the ~42 GB caching
        # phase), which fits a 48 GB Mac.
        self._text_encoder = None
        mx.clear_cache()
        return {"itemCount": count, "resolution": resolution, "latentEdge": latent_edge}

    def train_step(self, *, step: int, total_steps: int, config: TrainingRunConfig) -> float:
        mx = self._mx
        nn = importlib.import_module("mlx.nn")
        tree_utils = importlib.import_module("mlx.utils")
        transformer_mod = importlib.import_module("mlx_video.models.ltx.transformer")
        Modality = transformer_mod.Modality

        index = (step - 1) % len(self._latents)
        clean = self._latents[index]
        embeds = self._embeds[index]
        positions = self._positions

        # Rectified-flow training: x_t = (1 - sigma)*clean + sigma*noise, with the
        # raw transformer output regressed to v = noise - clean (ltx_flow_target).
        noise = mx.random.normal(clean.shape, dtype=clean.dtype)
        sigma = float(mx.random.uniform(low=1e-3, high=1.0 - 1e-3).item())
        x_t = (1.0 - sigma) * clean + sigma * noise
        target = ltx_flow_target(clean, noise)
        timesteps = mx.full((clean.shape[0], clean.shape[1]), sigma, dtype=clean.dtype)

        def loss_fn(model: Any) -> Any:
            modality = Modality(
                latent=x_t,
                timesteps=timesteps,
                positions=positions,
                context=embeds,
                context_mask=None,
                enabled=True,
            )
            velocity, _ = model(video=modality, audio=None)
            return mx.mean((velocity.astype(mx.float32) - target.astype(mx.float32)) ** 2)

        loss_and_grad = nn.value_and_grad(self._transformer, loss_fn)
        loss, grads = loss_and_grad(self._transformer)

        accum = max(1, config.gradient_accumulation)
        if self._accumulated_grads is None:
            self._accumulated_grads = grads
        else:
            self._accumulated_grads = tree_utils.tree_map(
                lambda a, b: a + b, self._accumulated_grads, grads
            )
        if step % accum == 0 or step == total_steps:
            averaged = tree_utils.tree_map(lambda g: g / float(accum), self._accumulated_grads)
            self._optimizer.update(self._transformer, averaged)
            mx.eval(self._transformer.parameters(), self._optimizer.state)
            self._accumulated_grads = None
        else:
            mx.eval(self._accumulated_grads)
        return float(loss)

    def _save_lora(self, *, output_dir: str, file_name: str) -> str:
        """Write the trained LoRA as safetensors keyed by the real LTX module
        paths (``{module}.lora_A.weight`` [rank,in] / ``{module}.lora_B.weight``
        [out,rank] + scalar ``{module}.alpha``), so ``mlx_video.lora`` round-trips
        with no key remap and reproduces the same delta (scale = alpha/rank)."""
        mx = self._mx
        os.makedirs(output_dir, exist_ok=True)
        state: dict[str, Any] = {}
        for path in self._lora_paths:
            wrapper = _get_submodule(self._transformer, path)
            rank = wrapper.lora_a.shape[0]
            state[f"{path}.lora_A.weight"] = wrapper.lora_a.astype(mx.float32)
            state[f"{path}.lora_B.weight"] = wrapper.lora_b.astype(mx.float32)
            state[f"{path}.alpha"] = mx.array(float(wrapper.scale) * float(rank), dtype=mx.float32)
        mx.eval(list(state.values()))
        output_path = os.path.join(output_dir, file_name)
        mx.save_safetensors(output_path, state)
        return output_path

    def save_checkpoint(self, *, step: int, output_dir: str, file_name: str) -> str | None:
        stem = Path(file_name).stem or "lora"
        return self._save_lora(output_dir=output_dir, file_name=f"{stem}-step{step:06d}.safetensors")

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
        """Render a short clip from the in-progress LoRA at each ``sampleEvery``
        checkpoint so training progress is visible — the flow-matching loss is
        uninformative (see :func:`ltx_flow_target`), so a generated preview is the
        only honest signal of whether the adapter is converging.

        The trained transformer carries the live adapter, but the generation
        pipeline builds its own model and applies a *saved* LoRA, so we save the
        current weights and drive the real ``generate_video_with_audio`` path with
        them applied — previews then match final inference exactly. This reloads
        the full inference stack (transformer + gemma + VAE decoder) per call, so
        it is memory- and time-heavy and only runs when ``sampleEvery > 0``. A
        preview failure is swallowed so it never aborts the training run.
        """
        cleaned = [str(p).strip() for p in (prompts or []) if str(p).strip()]
        if not cleaned or not self._lora_paths or self._transformer is None:
            return []

        mx = self._mx
        target = plan.get("target") or {}
        advanced = config.advanced if isinstance(config.advanced, dict) else {}
        repo = str(target.get("baseModelRepo") or LTX_MLX_REPO)
        text_repo = str(advanced.get("textEncoderRepo") or LTX_MLX_TEXT_ENCODER_REPO)

        # Fixed, modest preview geometry (divisible by 64) and a fixed seed so the
        # same prompt is directly comparable across checkpoints — only the adapter
        # changes between previews.
        edge = max(256, min(512, (bucket_resolution(config.resolution) // 64) * 64))
        num_frames = 25
        seed = int(config.seed)

        sample_dir = Path(output_dir) / "samples" / f"step-{step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_name).stem or "lora"

        # Persist the in-progress adapter where the inference loader can ingest it.
        tmp_lora = self._save_lora(
            output_dir=str(sample_dir), file_name=f".{stem}-adapter.safetensors"
        )

        generate_av = importlib.import_module("mlx_video.generate_av")
        lora_mod = importlib.import_module("mlx_video.lora")
        video_adapters = importlib.import_module("scene_worker.video_adapters")
        video_adapters._ensure_ffmpeg_on_path()
        video_adapters._install_ltx_lora_patch()

        samples: list[dict[str, Any]] = []
        for index, prompt in enumerate(cleaned):
            sample_path = sample_dir / f"{stem}-step{step:06d}-{index + 1}.mp4"
            try:
                module_map = lora_mod.load_multiple_loras(
                    [lora_mod.LoRAConfig(path=Path(tmp_lora), strength=1.0)]
                )
                token = video_adapters._PENDING_LTX_LORAS.set(module_map)
                try:
                    with _silence_output_fds():
                        generate_av.generate_video_with_audio(
                            model_repo=repo,
                            text_encoder_repo=text_repo,
                            prompt=prompt,
                            height=edge,
                            width=edge,
                            num_frames=num_frames,
                            seed=seed,
                            output_path=str(sample_path),
                            cfg_scale=config.sample_guidance_scale,
                            num_inference_steps=max(2, config.sample_steps),
                            enhance_prompt=False,
                            no_audio=True,
                            verbose=False,
                        )
                finally:
                    video_adapters._PENDING_LTX_LORAS.reset(token)
            except Exception as exc:  # noqa: BLE001 — a preview must never abort training
                emit_worker_event(
                    "training_sample_render_failed",
                    kernel=LtxMlxLoraTrainer.kernel_id,
                    step=step,
                    error=str(exc),
                )
                continue
            finally:
                if mx is not None:
                    with contextlib.suppress(Exception):
                        mx.clear_cache()
            if sample_path.exists():
                samples.append(
                    {
                        "step": step,
                        "prompt": prompt,
                        "path": str(sample_path),
                        "relativePath": project_relative_path(plan, sample_path),
                        "sampleSource": "live_adapter",
                        "mediaType": "video",
                        "numInferenceSteps": config.sample_steps,
                        "guidanceScale": config.sample_guidance_scale,
                        "createdAt": now(),
                    }
                )

        with contextlib.suppress(OSError):
            os.remove(tmp_lora)
        return samples

    def save_final(self, *, output_dir: str, file_name: str) -> str:
        return self._save_lora(output_dir=output_dir, file_name=file_name)

    def cleanup(self) -> None:
        self._latents = []
        self._embeds = []
        self._positions = None
        self._transformer = None
        self._vae = None
        self._text_encoder = None
        self._loaded_source = None
        mx = self._mx
        if mx is not None:
            try:
                mx.clear_cache()
            except Exception:
                pass


class LtxMlxLoraTrainer(ZImageLoraTrainer):
    """LTX-2.3 video LoRA trainer (native MLX, Apple Silicon).

    Reuses :class:`ZImageLoraTrainer`'s backend-agnostic staged orchestration
    (prepare → load → cache → train → checkpoint → save) with a native MLX
    backend. Trained from a still-image dataset; the output is an ``ltx-video``
    family LoRA the MLX LTX adapter loads at inference.
    """

    kernel_id = "ltx_mlx_lora"

    def _create_backend(self) -> _LtxMlxLoraBackend:
        return _LtxMlxLoraBackend()


# --------------------------------------------------------------------------- #
# Kernel registry
# --------------------------------------------------------------------------- #


_TRAINING_KERNELS: dict[str, Callable[[], Any]] = {
    ZImageLoraTrainer.kernel_id: ZImageLoraTrainer,
    LtxMlxLoraTrainer.kernel_id: LtxMlxLoraTrainer,
}


def create_training_kernel(kernel_id: Any) -> ZImageLoraTrainer:
    """Return the trainer for a plan's ``target.kernel`` id, or raise for an
    unknown kernel. Mirrors the image/video adapter factories."""

    factory = _TRAINING_KERNELS.get(str(kernel_id or "").strip())
    if factory is None:
        known = ", ".join(sorted(_TRAINING_KERNELS)) or "(none)"
        raise TrainingKernelError(
            f"No training kernel for {kernel_id!r}. Known kernels: {known}."
        )
    return factory()
