"""Out-of-process Lens LoRA training runner.

Executed by `scene_worker.training_adapters.LensLoraTrainer` via the dedicated
Lens sidecar venv (/opt/lens-venv) — NOT the main worker venv. Lens needs
transformers 5.x + diffusers 0.38, which conflict with the main worker stack
(transformers 4.x) that the in-process `ZImageLoraTrainer` runs in. So, like the
inference sidecar (`lens_runner.py`), the *entire* training loop runs here in one
subprocess; per-step IPC across the venv boundary would be far too chatty.

Contract: argv[1] is a path to a JSON spec the driver writes:
    {
      "source": "<repo id or local dir loadable by LensPipeline.from_pretrained>",
      "device": "cuda", "dtype": "bfloat16", "disableMxfp4": false,
      "outputDir": "<abs>", "fileName": "name.safetensors",
      "config": { ...TrainingRunConfig as a dict... },
      "items": [{"imagePath": "<abs>", "caption": "..."}, ...],
      "samplePrompts": ["..."],
      "progressPath": "<work>/progress.jsonl",
      "resultPath": "<work>/result.json"
    }
The runner appends JSONL progress events to ``progressPath`` (the driver tails
them and maps them onto the worker's progress bands) and writes a single result
object to ``resultPath`` on success. Diagnostics go to stderr (the worker log).
A non-zero exit with an ``{"error": ...}`` result signals failure.

Two Lens-specific details that diverge from the Z-Image trainer and are
load-bearing (see the inline notes):
  * The flow-matching velocity target is ``noise - latents`` (Lens feeds the
    transformer output to the scheduler *without* negation, unlike Z-Image).
  * There is no VAE encode path on the pipeline, so latents are produced by
    inverting ``LensPipeline._decode`` exactly.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


def _force_utf8_stdio() -> None:
    """Force stdout/stderr to UTF-8 before importing transformers. On Windows the
    sidecar's streams default to cp1252, and transformers' ``@auto_docstring``
    decorator prints a 🚨 emoji while decorating model classes at import, which
    crashes the process with UnicodeEncodeError. UTF-8 is already the default on
    Linux (where the worker container runs), so this is a no-op there. The runner's
    stdout IPC stays ASCII (json.dumps ensure_ascii), so widening never changes the
    result bytes. Mirrors scene_worker.runtime._force_utf8_stdio for the sidecar.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _log(message: str) -> None:
    sys.stderr.write(f"[lens_train_runner] {message}\n")
    sys.stderr.flush()


class _Progress:
    """Append-only JSONL progress sink the driver tails."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def emit(self, **event: Any) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
                handle.flush()
        except OSError:
            pass


# Report a training tick at most this often (in steps); the final step always
# reports. Mirrors training_adapters.PROGRESS_STEP_INTERVAL.
PROGRESS_STEP_INTERVAL = 10
DEFAULT_LORA_TARGET_MODULES = ["img_qkv", "txt_qkv", "to_out", "to_add_out"]


def _read_config(spec: dict[str, Any]) -> dict[str, Any]:
    config = spec.get("config")
    if not isinstance(config, dict):
        raise ValueError("Lens training spec is missing a 'config' object.")
    return config


def _advanced(config: dict[str, Any]) -> dict[str, Any]:
    advanced = config.get("advanced")
    return advanced if isinstance(advanced, dict) else {}


def _target_modules(config: dict[str, Any]) -> list[str]:
    modules = config.get("lora_target_modules") or _advanced(config).get("loraTargetModules")
    if isinstance(modules, str):
        return [token.strip() for token in modules.split(",") if token.strip()]
    if isinstance(modules, (list, tuple)) and modules:
        return [str(token) for token in modules]
    return list(DEFAULT_LORA_TARGET_MODULES)


def _select_dtype(torch: Any, device: str, mixed_precision: Any) -> Any:
    token = str(mixed_precision or "").strip().lower()
    if token in {"fp16", "float16", "half"}:
        return torch.float16
    if token in {"fp32", "float32", "full"}:
        return torch.float32
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


def _build_optimizer(torch: Any, name: str, params: list[Any], lr: float, weight_decay: float) -> Any:
    normalized = (name or "adamw").strip().lower()
    if normalized == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            _log("bitsandbytes unavailable; falling back to torch.optim.AdamW for adamw8bit.")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if normalized == "prodigyopt":
        try:
            from prodigyopt import Prodigy  # type: ignore

            return Prodigy(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            _log("prodigyopt unavailable; falling back to torch.optim.AdamW.")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if normalized in {"rose", "rose-opt", "rose_opt", "roseopt"}:
        try:
            from rose_opt import Rose  # type: ignore

            # compute_dtype="fp32": Rose defaults to fp64 internally, which has
            # no MPS kernel; fp32 is safe on both CUDA and Apple Silicon.
            return Rose(params, lr=lr, weight_decay=weight_decay, compute_dtype="fp32")
        except Exception:
            _log("rose-opt unavailable; falling back to torch.optim.AdamW.")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if normalized == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _lr_decay_multiplier(name: str, step: int, total: int, warmup: int) -> float:
    """Base-LR multiplier in [0, 1] at optimizer-update ``step`` (0-indexed),
    byte-identical to training_adapters.lr_decay_multiplier."""

    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup + 1)
    if total <= warmup:
        return 1.0
    progress = min(1.0, max(0.0, float(step - warmup) / float(total - warmup)))
    if name == "linear":
        return 1.0 - progress
    if name == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


def _build_lr_scheduler(torch: Any, optimizer: Any, name: str, total_updates: int, warmup_updates: int) -> Any:
    normalized = (name or "constant").strip().lower().replace("-", "_")
    if normalized not in {"constant", "linear", "cosine"}:
        raise ValueError(f"Unsupported lrScheduler '{name}'. Use constant, linear, or cosine.")
    total = max(1, int(total_updates))
    warmup = max(0, min(int(warmup_updates), total - 1))
    if normalized == "constant" and warmup == 0:
        return None

    def lr_lambda(step: int) -> float:
        return _lr_decay_multiplier(normalized, step, total, warmup)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _seeded_sample(torch: Any, fn: Any, shape: Any, *, generator: Any, device: str, dtype: Any) -> Any:
    """Draw seeded random values, MPS-safe.

    ``torch.Generator`` only lives on cpu/cuda, so on Apple Silicon a seeded run
    pairs a cpu generator with tensors on ``mps``. ``torch.randn``/``torch.rand``
    reject a cpu generator alongside a non-cpu ``device=`` argument, so when the
    generator's device differs from the target we draw on the generator's device
    and move — mirroring diffusers' ``randn_tensor`` and the main worker's
    ``training_adapters.seeded_sample``. ``fn`` is ``torch.randn`` or ``torch.rand``.
    """
    if generator is not None and generator.device.type != torch.device(device).type:
        return fn(shape, generator=generator, device=generator.device, dtype=dtype).to(device)
    return fn(shape, generator=generator, device=device, dtype=dtype)


def _sample_timestep(torch: Any, generator: Any, device: str, dtype: Any, ts_type: str, ts_bias: str) -> Any:
    """Sample a normalized flow-matching timestep (the noise fraction) in [0, 1],
    matching training_adapters.sample_training_timestep."""

    normalized_type = (ts_type or "sigmoid").strip().lower().replace("-", "_")
    if normalized_type in {"linear", "uniform"}:
        t = _seeded_sample(torch, torch.rand, 1, generator=generator, device=device, dtype=dtype)
    elif normalized_type == "weighted":
        base = _seeded_sample(torch, torch.rand, 1, generator=generator, device=device, dtype=dtype)
        center = torch.sigmoid(
            _seeded_sample(torch, torch.randn, 1, generator=generator, device=device, dtype=dtype)
        )
        t = (base + center) / 2.0
    else:
        t = torch.sigmoid(
            _seeded_sample(torch, torch.randn, 1, generator=generator, device=device, dtype=dtype)
        )

    normalized_bias = (ts_bias or "balanced").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_bias in {"high", "high_noise", "favor_high_noise"}:
        t = torch.sqrt(t)
    elif normalized_bias in {"low", "low_noise", "favor_low_noise"}:
        t = t * t
    return t.clamp(1e-3, 1.0 - 1e-3)


def _training_loss(torch: Any, prediction: Any, target: Any, loss_type: str) -> Any:
    normalized = (loss_type or "mse").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"mae", "l1", "mean_absolute_error"}:
        return torch.nn.functional.l1_loss(prediction.float(), target.float())
    return torch.nn.functional.mse_loss(prediction.float(), target.float())


def _bucket_resolution(value: int) -> int:
    """Snap a square edge to a multiple of the VAE scale factor (16)."""

    edge = max(256, int(value or 1024))
    return edge - (edge % 16)


def _load_pixel(torch: Any, image_path: str, resolution: int, dtype: Any, device: str) -> Any:
    """Load an image, center-crop-resize to a square ``resolution`` edge, and
    normalize to a [-1, 1] tensor [1, 3, H, W] — the inverse of the pipeline's
    ``_to_pil`` ([-1, 1] -> [0, 255])."""

    from PIL import Image

    edge = _bucket_resolution(resolution)
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        scale = edge / float(min(width, height))
        resized = image.resize(
            (max(edge, round(width * scale)), max(edge, round(height * scale))),
            Image.LANCZOS,
        )
        left = (resized.width - edge) // 2
        top = (resized.height - edge) // 2
        cropped = resized.crop((left, top, left + edge, top + edge))

    import numpy as np

    array = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device, dtype=dtype)


def _encode_latents(torch, lens_pipeline_cls, pipe: Any, pixel: Any, generator: Any) -> Any:
    """Encode a pixel tensor [1, 3, H, W] to a Lens training latent [1, S, 128].

    There is no encode path on ``LensPipeline``; this inverts ``_decode``
    (pipeline.py) exactly. ``_decode`` is, in order:
        latents[1,S,128] --rearrange--> lat1[1,32,H1,W1] --_patchify--> xp[1,128,h,w]
        --(xp/scale - shift)--> x1 --_unpatchify--> x2[1,32,H1,W1] --vae.decode-->
    so the inverse is vae.encode then re-apply the same bn normalization the other
    way (``xp = (x1 + shift) * scale``) and the inverse rearrange.
    """
    from einops import rearrange

    vae = pipe.vae
    z = vae.encode(pixel.to(vae.dtype)).latent_dist.sample(generator=generator)  # [1, 32, H1, W1]

    bn = vae.bn
    mean = bn.running_mean.view(1, -1, 1, 1)
    var = bn.running_var.view(1, -1, 1, 1)
    std = torch.sqrt(var + vae.config.batch_norm_eps)
    shift = (-mean).to(device=z.device, dtype=z.dtype)
    scale = (1.0 / std).to(device=z.device, dtype=z.dtype)

    xp = lens_pipeline_cls._patchify_latents(z)  # [1, 128, h, w]
    xp = (xp + shift) * scale  # inverse of decode's ``x / scale - shift``
    lat1 = lens_pipeline_cls._unpatchify_latents(xp)  # [1, 32, H1, W1]
    latents = rearrange(lat1, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=2, p2=2)
    return latents.contiguous()


def _save_lora(transformer: Any, output_dir: str, file_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    # LensPipeline has no LoRA loader mixin, but LensTransformer2DModel inherits
    # diffusers' PeftAdapterMixin, so save/load the adapter directly on it. The
    # inference path (sc-1587) loads the same file with ``load_lora_adapter``.
    transformer.save_lora_adapter(output_dir, weight_name=file_name, safe_serialization=True)
    return str(Path(output_dir) / file_name)


def train(spec: dict[str, Any], progress: _Progress) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

    import torch  # noqa: E402
    import transformers  # noqa: E402
    import peft  # noqa: E402
    from lens import LensGptOssEncoder, LensPipeline  # noqa: E402

    config = _read_config(spec)
    advanced = _advanced(config)
    source = str(spec["source"])
    output_dir = str(spec["outputDir"])
    file_name = str(spec.get("fileName") or "lora.safetensors")
    items = list(spec.get("items") or [])
    if not items:
        raise ValueError("Lens training spec has no dataset items.")

    device = str(spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Lens training requested a CUDA device but torch.cuda.is_available() is False in the "
            "lens venv. Rebuild the worker image with a CUDA (cu128) torch in /opt/lens-venv."
        )
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    dtype = _select_dtype(torch, device, advanced.get("mixedPrecision"))
    disable_mxfp4 = bool(spec.get("disableMxfp4", False)) or not device.startswith("cuda")

    steps = max(1, int(config.get("steps") or 1))
    rank = int(config.get("rank") or 16)
    alpha = int(config.get("alpha") or rank)
    learning_rate = float(config.get("learning_rate") or 1e-4)
    weight_decay = float(config.get("weight_decay") or 0.0)
    grad_accum = max(1, int(config.get("gradient_accumulation") or 1))
    resolution = int(config.get("resolution") or 1024)
    seed = int(config.get("seed") or 0)
    save_every = int(config.get("save_every") or 0)
    sample_every = int(config.get("sample_every") or 0)
    sample_steps = int(config.get("sample_steps") or 4)
    sample_guidance = float(config.get("sample_guidance_scale") or 1.0)
    sample_prompts = [str(p) for p in (spec.get("samplePrompts") or config.get("sample_prompts") or []) if str(p).strip()]
    ts_type = str(advanced.get("timestepType") or config.get("timestep_type") or "sigmoid")
    ts_bias = str(advanced.get("timestepBias") or config.get("timestep_bias") or "balanced")
    loss_type = str(advanced.get("lossType") or config.get("loss_type") or "mse")
    lr_scheduler_name = str(advanced.get("lrScheduler") or config.get("lr_scheduler") or "constant")
    lr_warmup_steps = int(advanced.get("lrWarmupSteps") or config.get("lr_warmup_steps") or 0)
    gradient_checkpointing = bool(advanced.get("gradientCheckpointing", config.get("gradient_checkpointing", True)))
    target_modules = _target_modules(config)

    _log(
        f"torch {torch.__version__} transformers {transformers.__version__} device={device} dtype={dtype} "
        f"steps={steps} rank={rank} alpha={alpha} lr={learning_rate} targets={target_modules}"
    )

    # ---- Load the base model (microsoft/Lens by default) -------------------
    progress.emit(event="stage", stage="loading_model", message=f"Loading Lens base model ({source}).")
    text_encoder_kwargs: dict[str, Any] = {"subfolder": "text_encoder", "dtype": dtype}
    mxfp4_config = getattr(transformers, "Mxfp4Config", None)
    if mxfp4_config is not None:
        text_encoder_kwargs["quantization_config"] = mxfp4_config(dequantize=disable_mxfp4)
    text_encoder = LensGptOssEncoder.from_pretrained(source, **text_encoder_kwargs)
    pipe = LensPipeline.from_pretrained(source, text_encoder=text_encoder, torch_dtype=dtype)
    pipe.to(device)

    transformer = pipe.transformer
    transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    generator = torch.Generator("cpu").manual_seed(seed)

    # ---- Cache latents + 4-layer text features -----------------------------
    progress.emit(event="stage", stage="caching_latents", message=f"Encoding {len(items)} dataset item(s).")
    cached: list[dict[str, Any]] = []
    latent_h = _bucket_resolution(resolution) // pipe.vae_scale_factor
    latent_w = latent_h
    with torch.no_grad():
        for index, item in enumerate(items):
            image_path = str(item.get("imagePath") or "")
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Dataset item {index} image is missing: {image_path!r}")
            pixel = _load_pixel(torch, image_path, resolution, dtype, device)
            latents = _encode_latents(torch, LensPipeline, pipe, pixel, generator=None)
            # encode_prompt returns (features[list of 4], mask, neg_features, neg_mask);
            # training is single-conditional, so keep only the positives.
            features, mask, _, _ = pipe.encode_prompt(
                prompt=str(item.get("caption") or ""),
                negative_prompt="",
                num_images_per_prompt=1,
                device=torch.device(device),
            )
            cached.append(
                {
                    "latents": latents.detach().to("cpu"),
                    "features": [feat.detach().to("cpu") for feat in features],
                    "mask": mask.detach().to("cpu"),
                }
            )
            progress.emit(event="cache", done=index + 1, total=len(items))
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps" and getattr(torch, "mps", None) is not None:
        # Release the caching-phase allocator cache; the Lens stack peaks near
        # ~90 GB on Apple Silicon, so reclaiming here eases the train-loop ceiling.
        torch.mps.empty_cache()

    # ---- Attach the trainable LoRA -----------------------------------------
    progress.emit(event="stage", stage="loading_model", message="Attaching LoRA adapter to the transformer.")
    lora_config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
    )
    transformer.add_adapter(lora_config)
    for method_name in ("set_adapter", "enable_adapters"):
        method = getattr(transformer, method_name, None)
        if method is None:
            continue
        try:
            method("default") if method_name == "set_adapter" else method()
            break
        except Exception as exc:  # noqa: BLE001
            _log(f"LoRA activation via {method_name} failed: {exc}")

    if gradient_checkpointing:
        if hasattr(transformer, "enable_input_require_grads"):
            try:
                transformer.enable_input_require_grads()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        elif hasattr(transformer, "gradient_checkpointing_enable"):
            transformer.gradient_checkpointing_enable()

    transformer.train()
    trainable = [param for param in transformer.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError(
            "LoRA adapter attached no trainable parameters; the target modules "
            f"{target_modules} matched no LensTransformer2DModel layers. Lens uses fused "
            "QKV (img_qkv/txt_qkv) plus to_out/to_add_out — adjust advanced.loraTargetModules."
        )
    _log(f"trainable LoRA tensors: {len(trainable)}")

    optimizer = _build_optimizer(torch, str(config.get("optimizer") or "adamw"), trainable, learning_rate, weight_decay)
    optimizer.zero_grad()
    total_updates = max(1, (steps + grad_accum - 1) // grad_accum)
    warmup_updates = (max(0, lr_warmup_steps) + grad_accum - 1) // grad_accum
    warmup_updates = max(0, min(warmup_updates, total_updates - 1))
    lr_scheduler = _build_lr_scheduler(torch, optimizer, lr_scheduler_name, total_updates, warmup_updates)

    train_generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed + 1)
    img_shapes = [(1, latent_h, latent_w)]

    # ---- Training loop ------------------------------------------------------
    progress.emit(event="stage", stage="training", message=f"Training for {steps} step(s).")
    checkpoints: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    completed_steps = 0
    for step in range(1, steps + 1):
        entry = cached[(step - 1) % len(cached)]
        latents = entry["latents"].to(device=device, dtype=dtype)
        features = [feat.to(device=device, dtype=dtype) for feat in entry["features"]]
        mask = entry["mask"].to(device)

        noise = _seeded_sample(
            torch, torch.randn, latents.shape, generator=train_generator, device=device, dtype=dtype
        )
        t = _sample_timestep(torch, train_generator, device, dtype, ts_type, ts_bias)
        noisy = (1.0 - t) * latents + t * noise
        # Lens feeds the transformer output to FlowMatchEulerDiscreteScheduler.step
        # WITHOUT negation (pipeline.py), so the scheduler integrates the model
        # output as the velocity ``noise - latents``. The transformer timestep is
        # the noise fraction ``t`` in [0, 1] directly (no inversion). This is the
        # opposite sign from the Z-Image trainer, which trains ``latents - noise``.
        target = noise - latents
        model_timestep = t.reshape(1).to(dtype)

        model_out = transformer(
            hidden_states=noisy,
            encoder_hidden_states=features,
            encoder_hidden_states_mask=mask,
            timestep=model_timestep,
            img_shapes=img_shapes,
        )
        if isinstance(model_out, (list, tuple)):
            model_out = model_out[0]

        loss = _training_loss(torch, model_out, target, loss_type)
        (loss / grad_accum).backward()
        if step % grad_accum == 0 or step == steps:
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
        completed_steps = step

        loss_value = float(loss.detach().to("cpu"))
        if step == steps or step % PROGRESS_STEP_INTERVAL == 0:
            progress.emit(event="step", step=step, total=steps, loss=loss_value)

        if save_every and step % save_every == 0 and step < steps:
            stem = Path(file_name).stem or "lora"
            checkpoint_name = f"{stem}-step{step:06d}.safetensors"
            progress.emit(event="stage", stage="checkpointing", step=step, message=f"Saving checkpoint at step {step}.")
            checkpoint_path = _save_lora(transformer, output_dir, checkpoint_name)
            checkpoints.append({"step": step, "path": checkpoint_path})

        if sample_every and sample_prompts and step % sample_every == 0:
            try:
                rendered = _render_samples(
                    torch, pipe, transformer, sample_prompts, output_dir, file_name,
                    step=step, resolution=resolution, sample_steps=sample_steps,
                    guidance_scale=sample_guidance, seed=seed,
                )
                if rendered:
                    samples.extend(rendered)
                    progress.emit(event="sample", step=step, samples=rendered)
            except Exception as exc:  # noqa: BLE001 - samples are best-effort previews
                _log(f"sample render at step {step} failed (continuing): {exc}")

    # ---- Save final adapter -------------------------------------------------
    progress.emit(event="stage", stage="saving", message="Saving trained LoRA weights.")
    output_path = _save_lora(transformer, output_dir, file_name)
    progress.emit(event="saved", path=output_path)

    return {
        "outputPath": output_path,
        "fileName": file_name,
        "outputDir": output_dir,
        "steps": steps,
        "stepsCompleted": completed_steps,
        "checkpoints": checkpoints,
        "trainingSamples": samples,
        "rank": rank,
        "alpha": alpha,
        "learningRate": learning_rate,
        "resolution": _bucket_resolution(resolution),
        "loraTargetModules": list(target_modules),
        "baseModelSource": source,
    }


def _render_samples(
    torch: Any,
    pipe: Any,
    transformer: Any,
    prompts: list[str],
    output_dir: str,
    file_name: str,
    *,
    step: int,
    resolution: int,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Best-effort in-training previews, rendered on the loaded base model. Note
    these render through the base (multi-step) Lens, not Lens-Turbo; treat them
    as concept-progress previews, not deployment-fidelity samples."""

    sample_dir = Path(output_dir) / "samples" / f"step-{step:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_name).stem or "lora"
    edge = min(768, _bucket_resolution(resolution))
    was_training = bool(getattr(transformer, "training", False))
    transformer.eval()
    rendered: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for index, prompt in enumerate(prompts[:4]):
                generator = torch.Generator("cpu").manual_seed(seed + step + index)
                output = pipe(
                    prompt=prompt,
                    height=edge,
                    width=edge,
                    num_inference_steps=max(1, sample_steps),
                    guidance_scale=guidance_scale,
                    generator=generator,
                    enable_reasoner=False,
                )
                image = output.images[0].convert("RGB")
                sample_path = sample_dir / f"{stem}-step{step:06d}-{index + 1}.png"
                image.save(sample_path, "PNG")
                rendered.append({"step": step, "prompt": prompt, "path": str(sample_path)})
    finally:
        if was_training:
            transformer.train()
    return rendered


def main() -> int:
    _force_utf8_stdio()
    if len(sys.argv) != 2:
        print(json.dumps({"error": "lens_train_runner expects exactly one argument: the spec JSON path"}))
        return 2
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    progress = _Progress(Path(spec["progressPath"]))
    result_path = Path(spec.get("resultPath") or (spec_path.parent / "result.json"))

    started = time.time()
    result = train(spec, progress)
    result["elapsedSeconds"] = round(time.time() - started, 1)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({"outputPath": result.get("outputPath"), "stepsCompleted": result.get("stepsCompleted")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - surface any failure as structured JSON
        import traceback

        traceback.print_exc()
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            spec_arg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
            result_path = Path(spec_arg.get("resultPath") or (Path(sys.argv[1]).parent / "result.json"))
            result_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(payload))
        raise SystemExit(1)
