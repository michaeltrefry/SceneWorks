"""PuLID-FLUX face-identity adapter (sc-2012, epic 2003).

Zero-shot face-identity generation on FLUX.1-dev: an antelopev2 ArcFace embedding +
a custom EVA02-CLIP-L-14-336 visual encoder feed PuLID's IDFormer/PerceiverAttention
cross-attention adapter, which is injected into FLUX's DiT blocks via the bundled
BFL flow model.

Hardware spike (sc-2012, 2026-05-28): 0.8016 ArcFace cosine vs reference at id_weight=1.0,
timestep_to_start_cfg=4 ("PuLID photoreal" preset). MPS bf16 no-offload at 1024×1024 /
30 steps lands at ~127s/image and ~85 GB peak unified memory (so requires a 64 GB+
host; see manifest gating). Single MPS port carried in the vendored copy: `flux/math.py`
rope uses `float32` on MPS (no `float64` kernel on Metal).

Runs in the MAIN worker venv. Extra deps (see `requirements-pulid-flux.txt`): timm,
einops, ftfy, facexlib, insightface, onnxruntime, accelerate, sentencepiece. The
PuLID + BFL flow + EVA-CLIP code is vendored under `_vendor/pulid_flux/` (all
Apache-2.0); placed on sys.path here, mirroring `instantid_adapter` / `lens_runner`.

Reuses the shared sc-2009 antelopev2 helper (`_ensure_antelopev2`) so InstantID and
PuLID share one `~/.insightface/models/antelopev2/` install.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_adapters import (
    CancelCallback,
    ImageAssetWriter,
    ImageRequest,
    MODEL_TARGETS,
    ProgressCallback,
    activate_torch_device,
    emit_worker_event,
    huggingface_repo_cache_exists,
    image_batch_progress,
    format_batch_running_message,
    load_reference_image,
    release_inference_memory,
    require_inference_backend_for_gpu_worker,
    resolve_seed,
    safe_int,
    select_torch_device,
)
from .instantid_adapter import _ensure_antelopev2, _insightface_root
from .settings import WorkerSettings

_VENDOR = Path(__file__).resolve().parent / "_vendor" / "pulid_flux"

# PuLID-FLUX adapter weights (the IDFormer + PerceiverAttention cross-attention
# blocks); the FLUX-dev DiT itself comes from `flux_dev` MODEL_TARGETS["repo"].
_PULID_HF_REPO = "guozinan/PuLID"
_PULID_VERSION_DEFAULT = "v0.9.1"


def _require_pulid_flux_extras() -> None:
    """Fail fast with an actionable message when the optional PuLID-FLUX extras are
    not installed (rather than a raw `ModuleNotFoundError` mid-generation). timm +
    facexlib + ftfy drive EVA-CLIP and face alignment; insightface + onnxruntime
    overlap with InstantID and are usually already present. See
    `requirements-pulid-flux.txt`."""
    missing = [
        mod
        for mod in ("timm", "einops", "ftfy", "facexlib", "insightface", "onnxruntime")
        if importlib.util.find_spec(mod) is None
    ]
    if missing:
        raise RuntimeError(
            "PuLID-FLUX needs extra dependencies that are not installed in this worker "
            f"environment: {', '.join(missing)}. Install them with "
            "`pip install -r apps/worker/requirements-pulid-flux.txt`. In the desktop "
            "app, restart it to auto-provision the PuLID-FLUX extras."
        )


def _import_pulid_flux() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Import the vendored PuLID-FLUX + BFL flow + EVA-CLIP modules. Inserts
    `_vendor/pulid_flux/` on sys.path so the top-level package layout (`flux.*`,
    `pulid.*`, `eva_clip.*`) resolves identically to upstream."""
    vendor = str(_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    sampling = importlib.import_module("flux.sampling")
    util = importlib.import_module("flux.util")
    pipeline_flux = importlib.import_module("pulid.pipeline_flux")
    pulid_utils = importlib.import_module("pulid.utils")
    # Tell PuLIDPipeline where antelopev2 lives (shared with sc-2009 InstantID).
    os.environ["PULID_FLUX_INSIGHTFACE_ROOT"] = str(_insightface_root())
    return (
        sampling.denoise,
        sampling.get_noise,
        sampling.get_schedule,
        sampling.prepare,
        sampling.unpack,
        util.load_ae,
        util.load_clip,
        util.load_flow_model,
        util.load_t5,
    ), pipeline_flux.PuLIDPipeline, pulid_utils.resize_numpy_image_long


class PuLIDFluxAdapter:
    """FLUX.1-dev face-identity generation via PuLID-FLUX (IDFormer cross-attention).

    Maintains one loaded BFL flow model + PuLID encoder/CA across a batch (the
    weights cost ~37 GB to load); ID embedding extraction is per-call and runs in
    ~0.3–0.7 s once the antelopev2 stack is warm.
    """

    id = "pulid_flux"

    def __init__(self) -> None:
        # BFL flow model + AE + T5 + CLIP-L + PuLIDPipeline are loaded once per
        # worker session and reused across batches; loading costs ~32 s warm.
        self._flow_model: Any | None = None
        self._ae: Any | None = None
        self._t5: Any | None = None
        self._clip: Any | None = None
        self._pulid: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_model: str | None = None
        self._max_sequence_length: int = 128

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._loaded_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        if self._flow_model is None and self._pulid is None:
            return False
        self._flow_model = None
        self._ae = None
        self._t5 = None
        self._clip = None
        self._pulid = None
        self._loaded_repo = None
        self._loaded_model = None
        # gc.collect() BEFORE empty_cache() (release_inference_memory) is required so
        # the dropped ~37 GB PuLID-FLUX stack's reference cycles are freed immediately
        # rather than lingering beside the next model until a cyclic GC (sc-4192).
        release_inference_memory(importlib.import_module("torch"))
        return True

    # ---- model load ---------------------------------------------------
    def _load_pipeline(
        self,
        settings: WorkerSettings,
        request: ImageRequest,
        model_target: dict[str, Any],
        progress: ProgressCallback,
        *,
        job_id: str,
    ) -> None:
        """Load (or reuse the cached) BFL flow + AE + text encoders + PuLID stack."""
        torch = importlib.import_module("torch")
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device_str = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device_str)
        device = torch.device(device_str)

        if self._flow_model is not None and self._pulid is not None and self._loaded_repo == repo:
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            self._loaded_model = request.model
            return
        if self._flow_model is not None:
            self.unload()

        pulid_cfg = model_target.get("pulidFlux") or {}
        if not pulid_cfg:
            raise RuntimeError(f"{request.model} has no PuLID-FLUX configuration.")

        cache_action = "Loading cached" if huggingface_repo_cache_exists(repo) else "Downloading"
        progress("loading_model", "loading_model", 0.2, f"{cache_action} {model_target['label']} (PuLID-FLUX).")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device_str,
            dtype="bfloat16",
            usePuLIDFlux=True,
        )

        # Ensure antelopev2 is on disk BEFORE PuLIDPipeline.__init__ tries to load
        # it (shared with sc-2009 InstantID; the helper downloads on first run).
        _require_pulid_flux_extras()
        _ensure_antelopev2()

        helpers, pipeline_class, _resize = _import_pulid_flux()
        (_denoise, _get_noise, _get_schedule, _prepare, _unpack,
         load_ae, load_clip, load_flow_model, load_t5) = helpers
        # Vendored loaders default to device="cuda"; pass the resolved device string.
        max_seq_len = safe_int(
            request.advanced.get("maxSequenceLength"),
            int(pulid_cfg.get("maxSequenceLength", 128)),
            64,
            512,
        )
        self._max_sequence_length = max_seq_len
        self._t5 = load_t5(device_str, max_length=max_seq_len)
        self._clip = load_clip(device_str)
        # BFL config key on the upstream side (e.g. "flux-dev"); MODEL_TARGETS
        # carries `pulidFlux.bflConfig` to keep the wire-format readable.
        bfl_config = pulid_cfg.get("bflConfig", "flux-dev")
        self._flow_model = load_flow_model(bfl_config, device=device_str).eval()
        self._ae = load_ae(bfl_config, device=device_str)

        # onnx_provider="cpu" — no CUDAExecutionProvider on Mac, and we don't ship
        # the onnxruntime-gpu wheel anywhere. CPU runs antelopev2 in <1s/image.
        self._pulid = pipeline_class(
            self._flow_model,
            device=device,
            weight_dtype=torch.bfloat16,
            onnx_provider="cpu",
        )
        version = str(pulid_cfg.get("version", _PULID_VERSION_DEFAULT))
        self._pulid.load_pretrain(pretrain_path=None, version=version)

        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
        )
        self._loaded_repo = repo
        self._loaded_model = request.model

    # ---- knobs --------------------------------------------------------
    def _id_weight(self, request: ImageRequest) -> float:
        """PuLID's identity-strength knob (analog of InstantID's ip_adapter_scale).
        sc-2012 spike: 1.0 = baseline (0.8016 ArcFace), 0.8 starts to drift, 0.6 drifts
        further. Upstream UI range is 0.0–3.0 but useful band is 0.5–1.5."""
        try:
            value = float(request.advanced.get("idWeight", 1.0))
        except (TypeError, ValueError):
            return 1.0
        # Same clamp as the upstream gradio slider.
        return max(0.0, min(3.0, value))

    def _timestep_to_start_cfg(self, request: ImageRequest) -> int:
        """Higher = identity injected later in the denoise loop = more editable but
        weaker identity. Upstream guidance: 4 for photoreal, 0–1 for stylized. The
        sc-2012 photoreal default is 4 (matches the spike's best run)."""
        return safe_int(request.advanced.get("timestepToStartCfg"), 4, 0, 20)

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target.get("steps", 30), 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = model_target.get("guidanceScale", 4.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    # ---- inference ----------------------------------------------------
    def _run_pipeline(
        self,
        settings: WorkerSettings,
        request: ImageRequest,
        seed: int,
        project_path: Path,
        cancel_requested: CancelCallback | None = None,
    ) -> Image.Image:
        """One image: extract ID embedding from the reference (~0.3 s), run the
        FLUX DiT denoise with PuLID cross-attention injection, then AE-decode."""
        torch = importlib.import_module("torch")
        helpers, _PuLIDPipeline, resize_numpy_image_long = _import_pulid_flux()
        denoise, get_noise, get_schedule, prepare, unpack, *_ = helpers
        device_str = select_torch_device(torch, settings.gpu_id)
        device = torch.device(device_str)
        activate_torch_device(torch, device_str)

        # Reference → numpy (RGB, [0, 255]) → resize long side to 1024 (PuLID's
        # antelopev2 path expects ~1024-long input for stable face detection).
        reference = load_reference_image(project_path, request.reference_asset_id)
        id_image = resize_numpy_image_long(np.array(reference.convert("RGB")), 1024)

        model_target = MODEL_TARGETS[request.model]
        num_steps = self._num_inference_steps(request, model_target)
        guidance = self._guidance_scale(request, model_target)
        id_weight = self._id_weight(request)
        start_cfg = self._timestep_to_start_cfg(request)
        width, height = request.width, request.height

        # Noise + flow-matching schedule. Seed flows through BFL's get_noise so it
        # is reproducible across runs with the same (seed, width, height).
        with torch.inference_mode():
            x = get_noise(1, height, width, device=device, dtype=torch.bfloat16, seed=seed)
            timesteps = get_schedule(num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

            # T5 max_length tracks the per-call advanced.maxSequenceLength if the
            # cached value drifted (cheap reset; the encoder isn't re-loaded).
            self._t5.max_length = self._max_sequence_length
            inp = prepare(t5=self._t5, clip=self._clip, img=x, prompt=request.prompt)

            id_embeddings, _ = self._pulid.get_id_embedding(id_image, cal_uncond=False)
            if cancel_requested is not None and cancel_requested():
                raise InterruptedError("Image generation canceled by user.")

            # true_cfg=1.0 = "fake CFG" (PuLID's recommended default; the spike's
            # best baseline). The negative-prompt path (true_cfg > 1.0) is not
            # exposed yet — guidance + id_weight cover the photoreal preset.
            x = denoise(
                self._flow_model,
                **inp,
                timesteps=timesteps,
                guidance=guidance,
                id=id_embeddings,
                id_weight=id_weight,
                start_step=0,
                uncond_id=None,
                true_cfg=1.0,
                timestep_to_start_cfg=start_cfg,
                neg_txt=None,
                neg_txt_ids=None,
                neg_vec=None,
                aggressive_offload=False,
            )
            if cancel_requested is not None and cancel_requested():
                raise InterruptedError("Image generation canceled by user.")

            x = unpack(x.float(), height, width)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                x = self._ae.decode(x)

        x = x.clamp(-1, 1)
        # Match the BFL demo: rearrange CHW → HWC, scale [-1, 1] → [0, 255], PIL.
        from einops import rearrange  # noqa: PLC0415 — torchless deps live in the vendored extras
        x = rearrange(x[0], "c h w -> h w c")
        return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    def generate(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        model_target = MODEL_TARGETS.get(request.model, {})
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a PuLID-FLUX target.")
        if request.mode == "edit_image":
            raise RuntimeError(f"{request.model} does not support image editing.")
        if not request.reference_asset_id:
            raise RuntimeError("PuLID-FLUX generation requires a character reference image.")
        _require_pulid_flux_extras()

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']} (PuLID-FLUX).")
        self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = model_target["label"]
        total = request.count

        def image_at_index(index: int) -> Image.Image:
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            progress(
                "running",
                "generating",
                image_batch_progress(index, total),
                format_batch_running_message(label, index, total),
            )
            emit_worker_event(
                "image_inference_start",
                jobId=job["id"],
                adapter=self.id,
                model=request.model,
                imageIndex=index,
                imageCount=total,
                device=device,
            )
            try:
                image = self._run_pipeline(
                    settings, request, seed, project_path, cancel_requested=cancel_requested,
                )
            except Exception as exc:
                emit_worker_event(
                    "image_inference_failed",
                    jobId=job["id"],
                    adapter=self.id,
                    imageIndex=index,
                    error=str(exc),
                    errorType=exc.__class__.__name__,
                )
                raise
            emit_worker_event("image_inference_complete", jobId=job["id"], adapter=self.id, imageIndex=index)
            return image

        return ImageAssetWriter().write_incremental_outputs(
            request=request,
            project_path=project_path,
            image_count=total,
            image_at_index=image_at_index,
            adapter_id=self.id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings={
                **request.advanced,
                "repo": request.advanced.get("modelRepo") or model_target["repo"],
                "pulidFlux": True,
                "idWeight": self._id_weight(request),
                "timestepToStartCfg": self._timestep_to_start_cfg(request),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request, model_target),
                "maxSequenceLength": self._max_sequence_length,
                "realModelInference": True,
            },
            settings=settings,
            job_id=job["id"],
        )
