"""Out-of-process mflux (Apple MLX) image generation runner.

Executed by the sidecar-orchestrating adapters in
`scene_worker.image_adapters` (`MlxFluxAdapter`, `MlxQwenAdapter`, …) via the
dedicated mflux sidecar venv (`/opt/mlx-flux-venv`) — NOT the main worker venv.
mflux hard-requires transformers>=5 + huggingface_hub>=1, which conflict with
the main worker stack (transformers 4.57.x + huggingface_hub<1) that native
LTX-2.3 and the existing diffusers FluxPipeline / QwenImagePipeline paths
depend on. So mflux runs isolated here, mirroring the Lens sidecar pattern
(lens_runner.py / LensTurboAdapter).

The file is named `mlx_flux_runner.py` because FLUX was the first family wired
up (sc-1970); the runner is intentionally general across the mflux model
catalog (sc-1972 added Qwen-Image; Z-Image / FIBO / FLUX.2 follow the same
template). Adding a new mflux family is a one-arm extension to
``_resolve_model_handle``.

Contract: argv[1] is a path to a JSON spec; the runner writes one PNG per seed
into spec["outDir"] and prints a single result JSON object to stdout:
    {"images": ["<outDir>/mlx_<family>_0000.png", ...]}
Progress and diagnostics go to stderr (captured into the worker log). A non-zero
exit code with an "error" JSON on stdout signals failure to the adapter.

Spec keys:
    model: e.g. "flux_schnell" | "flux_dev" | "qwen_image" | "flux2_klein_9b"
        | "flux2_klein_9b_kv"
    prompt: str
    negativePrompt: str | None  (ignored by FLUX.2 — Flux2 disallows negatives;
        focus on describing what you want.)
    seeds: list[int]
    height: int
    width: int
    numInferenceSteps: int
    guidance: float
    quantize: int | None (None, 3, 4, 5, 6, 8)
    loras: list[{"path": str, "weight": float, "name": str}]
    imagePaths: list[str] | None  (FLUX.2 edit mode only — list of reference
        image paths. Triggers Flux2KleinEdit dispatch; for FLUX.2-klein-9b-kv
        the KV cache auto-engages.)
    outDir: str (sidecar writes PNGs + result.json here)

Validated 2026-05-28 against mflux 0.17.5 (sc-1969 FLUX spike, sc-1972 Qwen verify).
sc-2164 extends with FLUX.2-klein-9b / -kv via Flux2Klein / Flux2KleinEdit, pinned
to michaeltrefry/mflux@claude/flux2-kv-cache for the upstream KV-cache patch
(filipstrand/mflux#426).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _log(message: str) -> None:
    sys.stderr.write(f"[mlx_flux_runner] {message}\n")
    sys.stderr.flush()


def _resolve_model_handle(model_id: str, has_reference: bool) -> tuple[type, object, str, str]:
    """Map a SceneWorks model id + edit-vs-t2i hint onto an mflux
    (class, ModelConfig, filename_prefix, family) tuple.

    Each branch instantiates an mflux generation class for one model family and
    returns it alongside a `ModelConfig` factory and the per-image filename
    prefix used in `outDir`. The fourth tuple element is the family tag
    (``"flux1" | "qwen" | "z_image" | "flux2"``) the main loop uses to skip
    family-incompatible kwargs (e.g. ``negative_prompt`` is disallowed by
    Flux2). Keep this map in sync with the `_supported_models` sets in the
    corresponding adapters.

    ``has_reference`` selects the edit variant for families that have one
    (FLUX.2 → Flux2Klein vs Flux2KleinEdit). Families without an edit path
    ignore the flag.
    """
    from mflux.models.common.config.model_config import ModelConfig

    if model_id == "flux_schnell":
        from mflux.models.flux.variants.txt2img.flux import Flux1
        return Flux1, ModelConfig.schnell(), "mlx_flux", "flux1"
    if model_id == "flux_dev":
        from mflux.models.flux.variants.txt2img.flux import Flux1
        return Flux1, ModelConfig.dev(), "mlx_flux", "flux1"
    if model_id == "qwen_image":
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        return QwenImage, ModelConfig.qwen_image(), "mlx_qwen", "qwen"
    if model_id == "z_image_turbo":
        # NOTE: the ZImage import path is `variants.z_image`, not
        # `variants.txt2img.z_image` (mflux 0.17.5 — Z-Image hasn't been
        # refactored into the txt2img subpackage like Flux/Qwen). The
        # GeneratedImage return shape matches even though the type annotation
        # claims PIL.Image.Image — main() drops `.image` either way.
        from mflux.models.z_image.variants.z_image import ZImage
        return ZImage, ModelConfig.z_image_turbo(), "mlx_z_image", "z_image"
    if model_id == "flux2_klein_9b":
        if has_reference:
            from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
            return Flux2KleinEdit, ModelConfig.flux2_klein_9b(), "mlx_flux2_klein", "flux2"
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        return Flux2Klein, ModelConfig.flux2_klein_9b(), "mlx_flux2_klein", "flux2"
    if model_id == "flux2_klein_9b_kv":
        # The -kv variant only makes sense with a reference image (cache is
        # meaningless otherwise). MlxFlux2Adapter gates this in the main venv
        # so we get here only when has_reference is True.
        if not has_reference:
            raise RuntimeError(
                "mlx_flux_runner: flux2_klein_9b_kv requires a reference image; "
                "use flux2_klein_9b for text-to-image."
            )
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
        return Flux2KleinEdit, ModelConfig.flux2_klein_9b_kv(), "mlx_flux2_klein_kv", "flux2"
    raise RuntimeError(f"mlx_flux_runner: unsupported model id {model_id!r}.")


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "mlx_flux_runner expects exactly one argument: the spec JSON path"}))
        return 2
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    model_id = str(spec.get("model") or "")
    prompt = str(spec.get("prompt") or "")
    negative_prompt = spec.get("negativePrompt") or None
    seeds = [int(seed) for seed in spec.get("seeds") or []] or [0]
    # sc-2003 multi-backbone angle set: optional per-iteration prompt overrides
    # parallel to ``seeds``. None / absent / empty → all iterations use the
    # top-level ``prompt`` (the existing single-prompt batch path). When set,
    # the runner zips one prompt per seed and ignores the top-level ``prompt``
    # for that iteration. Adapter side (MlxFlux2Adapter) computes the augmented
    # prompts via character_studio_angles.augment_prompt_for_angle.
    prompts_override = spec.get("prompts") or None
    if prompts_override is not None and len(prompts_override) != len(seeds):
        raise RuntimeError(
            f"mlx_flux_runner: prompts list length ({len(prompts_override)}) "
            f"must equal seeds list length ({len(seeds)})."
        )
    height = int(spec.get("height") or 1024)
    width = int(spec.get("width") or 1024)
    steps = int(spec.get("numInferenceSteps") or 4)
    guidance = float(spec.get("guidance") or 0.0)
    quantize = spec.get("quantize")
    if quantize is not None:
        quantize = int(quantize)
    loras = spec.get("loras") or []
    image_paths = [str(p) for p in (spec.get("imagePaths") or []) if p]
    has_reference = bool(image_paths)
    out_dir = Path(spec["outDir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"

    # Heavy imports deferred until the spec is valid: a bad spec fails cleanly
    # before mflux loads MLX + the multi-GB transformer.
    model_cls, model_config, filename_prefix, family = _resolve_model_handle(model_id, has_reference)

    lora_paths: list[str] = []
    lora_scales: list[float] = []
    for index, lora in enumerate(loras):
        path = str(lora.get("path") or "")
        if not path:
            raise RuntimeError(f"mlx_flux_runner: LoRA #{index + 1} has no path.")
        try:
            scale = float(lora.get("weight", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        lora_paths.append(path)
        lora_scales.append(scale)

    _log(
        f"loading {model_cls.__name__} model={model_id} quantize={quantize} "
        f"loras={len(lora_paths)} steps={steps} guidance={guidance}"
    )
    model = model_cls(
        quantize=quantize,
        lora_paths=lora_paths or None,
        lora_scales=lora_scales or None,
        model_config=model_config,
    )
    _log(f"{model_cls.__name__} loaded; entering generation loop")

    images: list[str] = []
    for index, seed in enumerate(seeds):
        # mflux 0.17.5 generate_image() takes per-call kwargs; older 0.12.x
        # took a Config object. Pin in requirements-mlx-flux.txt anchors us
        # to the kwargs form. Per-family signature differences:
        #   - Flux2 (txt2img + edit) disallows negative_prompt entirely
        #     ("focus on describing what you want") and uses `image_paths` for
        #     reference list (not `image_path`).
        #   - Flux2Klein (txt2img) accepts a single optional `image_path`
        #     instead of a list; the runner only dispatches to it when there
        #     is no reference (image_paths is empty), so we don't pass it.
        #   - Other families (Flux1, Qwen, Z-Image) still take negative_prompt.
        # Per-iteration prompt override (sc-2003 angle set); falls back to the
        # top-level prompt when prompts_override isn't set.
        iter_prompt = prompts_override[index] if prompts_override else prompt
        gen_kwargs: dict[str, object] = {
            "seed": int(seed),
            "prompt": iter_prompt,
            "num_inference_steps": steps,
            "height": height,
            "width": width,
            "guidance": guidance,
        }
        if family != "flux2":
            gen_kwargs["negative_prompt"] = negative_prompt
        if family == "flux2" and has_reference:
            gen_kwargs["image_paths"] = image_paths
        result = model.generate_image(**gen_kwargs)
        path = out_dir / f"{filename_prefix}_{index:04d}.png"
        result.image.save(path, "PNG")
        images.append(str(path))
        _log(f"generated image {index + 1}/{len(seeds)} -> {path}")

    payload = {"images": images}
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(payload))
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
            out_dir = Path(spec_arg["outDir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(payload))
        raise SystemExit(1)
