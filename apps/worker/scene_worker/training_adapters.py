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
import importlib
import os
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
from .lora_adapters import huggingface_main_snapshot_dir, huggingface_snapshot_dirs
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
    mixed_precision: Any
    lora_target_modules: Any
    sample_every: int
    sample_steps: int
    sample_guidance_scale: float
    sample_prompts: list[str]
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
        mixed_precision=advanced.get("mixedPrecision"),
        lora_target_modules=target_modules,
        sample_every=_as_int(advanced.get("sampleEvery"), 0, minimum=0),
        sample_steps=_as_int(advanced.get("sampleSteps"), 9, minimum=1),
        sample_guidance_scale=_as_float(advanced.get("sampleGuidanceScale"), 0.0),
        sample_prompts=[str(prompt).strip() for prompt in sample_prompts if str(prompt).strip()][:4],
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

        progress("preparing", "preparing", 0.04, "Preparing Z-Image LoRA training run.")
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


def build_optimizer(name: str, params: list[Any], learning_rate: float) -> Any:
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
        return prodigy_module.Prodigy(params, lr=use_lr, eps=1e-6)
    if normalized in {"adamw8bit", "adam8bit"}:
        try:
            bnb = importlib.import_module("bitsandbytes")
            return bnb.optim.AdamW8bit(params, lr=learning_rate)
        except Exception:
            emit_worker_event(
                "training_optimizer_fallback",
                requested=name,
                using="adamw",
                reason="bitsandbytes unavailable",
            )
            return torch.optim.AdamW(params, lr=learning_rate)
    if normalized == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    return torch.optim.AdamW(params, lr=learning_rate)


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
        transformer.train()
        trainable = [param for param in transformer.parameters() if param.requires_grad]
        if not trainable:
            raise TrainingKernelError(
                "LoRA adapter attached no trainable parameters; the configured "
                "target modules did not match any transformer layers. Adjust "
                "advanced.loraTargetModules for this base model."
            )

        self._optimizer = build_optimizer(config.optimizer, trainable, config.learning_rate)
        self._optimizer.zero_grad()
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
        emit_worker_event(
            "training_pipeline_load_complete",
            kernel=ZImageLoraTrainer.kernel_id,
            source=source,
            trainableTensors=len(trainable),
            gpuMemory=gpu_memory_snapshot(torch, device),
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
        noise = torch.randn(
            latents.shape, generator=self._generator, device=device, dtype=latents.dtype
        )
        t = torch.rand(1, generator=self._generator, device=device, dtype=latents.dtype)
        t = t.clamp(1e-3, 1.0 - 1e-3)
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

        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        accum = max(1, config.gradient_accumulation)
        (loss / accum).backward()
        if step % accum == 0 or step == total_steps:
            self._optimizer.step()
            self._optimizer.zero_grad()
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
# Kernel registry
# --------------------------------------------------------------------------- #


_TRAINING_KERNELS: dict[str, Callable[[], Any]] = {
    ZImageLoraTrainer.kernel_id: ZImageLoraTrainer,
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
