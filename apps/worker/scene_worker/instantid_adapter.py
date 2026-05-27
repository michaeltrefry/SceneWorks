"""InstantID SDXL face-identity adapter (sc-2009).

Zero-shot face-identity generation: an insightface (antelopev2) ArcFace embedding
+ 5-point landmark ControlNet ("IdentityNet") preserve a person's identity from a
single reference while the prompt drives scene/pose/wardrobe. Validated in the
sc-2009 A/B as the only method that holds identity AND follows the prompt; plain
IP-Adapter (sc-2006/2007) only captures coarse resemblance.

Runs in the MAIN worker venv. Extra deps (see requirements-instantid.txt):
insightface, onnxruntime, onnx, peft, einops. The InstantX pipeline +
tencent-ailab ip_adapter module are vendored under _vendor/instantid (both
Apache-2.0); placed on sys.path here, mirroring lens_runner.

Key constraints (from the sc-2009 spike):
* bf16 on MPS — fp16 NaNs on Metal (the pipeline uses self.dtype throughout).
* The landmark control image MUST match the output aspect or faces stretch, so
  the reference is letterboxed onto a (width x height) canvas before face
  detection and draw_kps.
"""
from __future__ import annotations

import importlib
import importlib.util
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
    cancel_step_callback,
    emit_worker_event,
    filter_call_kwargs,
    image_batch_progress,
    format_batch_running_message,
    huggingface_repo_cache_exists,
    load_reference_image,
    require_inference_backend_for_gpu_worker,
    resolve_seed,
    safe_int,
    select_torch_device,
    select_torch_dtype,
)
from .settings import WorkerSettings

_VENDOR = Path(__file__).resolve().parent / "_vendor" / "instantid"

# antelopev2 face-analysis pack (insightface root/models/antelopev2/*.onnx). Not a
# standard insightface auto-download, so we mirror-fetch the 5 files on demand.
_ANTELOPEV2_REPO = "DIAMONIK7777/antelopev2"
_ANTELOPEV2_FILES = (
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx",
)


def _insightface_root() -> Path:
    import os

    return Path(os.environ.get("INSTANTID_INSIGHTFACE_ROOT", str(Path.home() / ".insightface")))


def _ensure_antelopev2() -> Path:
    """Make sure antelopev2 onnx models exist under the insightface root; return root."""
    from huggingface_hub import hf_hub_download

    root = _insightface_root()
    dest = root / "models" / "antelopev2"
    dest.mkdir(parents=True, exist_ok=True)
    for name in _ANTELOPEV2_FILES:
        if not (dest / name).exists():
            path = hf_hub_download(repo_id=_ANTELOPEV2_REPO, filename=name)
            # hf cache stores a symlink target; copy into the insightface layout.
            data = Path(path).read_bytes()
            (dest / name).write_bytes(data)
    return root


def _require_instantid_extras() -> None:
    """Fail fast with an actionable message when the optional InstantID extras are
    not installed (rather than a raw ModuleNotFoundError mid-generation). insightface
    + onnxruntime drive the face embedding/landmarks; einops is used by the vendored
    Resampler. See requirements-instantid.txt."""
    missing = [
        mod for mod in ("insightface", "onnxruntime", "einops") if importlib.util.find_spec(mod) is None
    ]
    if missing:
        raise RuntimeError(
            "InstantID needs extra dependencies that are not installed in this worker "
            f"environment: {', '.join(missing)}. Install them with "
            "`pip install -r apps/worker/requirements-instantid.txt`. In the desktop "
            "app, restart it to auto-provision the InstantID extras."
        )


def _import_instantid() -> tuple[Any, Any]:
    """Import the vendored InstantX pipeline (+ draw_kps). Inserts _vendor/instantid
    on sys.path so its top-level `from ip_adapter...` imports resolve."""
    vendor = str(_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    mod = importlib.import_module("pipeline_stable_diffusion_xl_instantid")
    return mod.StableDiffusionXLInstantIDPipeline, mod.draw_kps


def _letterbox(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize keeping aspect and pad onto a width x height canvas, so the landmark
    control image and the output share an aspect ratio (no face stretching)."""
    ratio = min(width / image.width, height / image.height)
    new_w, new_h = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


class InstantIDAdapter:
    """Identity-preserving SDXL generation via InstantID (face embedding + IdentityNet)."""

    id = "instantid_sdxl"

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_model: str | None = None
        self._face_app: Any | None = None

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._loaded_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        if self._pipe is None:
            return False
        self._pipe = None
        self._loaded_repo = None
        self._loaded_model = None
        self._empty_cache(importlib.import_module("torch"))
        return True

    @staticmethod
    def _empty_cache(torch: Any) -> None:
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- face analysis -------------------------------------------------
    def _face_analysis(self) -> Any:
        if self._face_app is None:
            root = _ensure_antelopev2()
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="antelopev2", root=str(root), providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._face_app = app
        return self._face_app

    def _largest_face(self, canvas: Image.Image) -> Any:
        import cv2

        bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        faces = self._face_analysis().get(bgr)
        if not faces:
            raise RuntimeError(
                "No face detected in the reference image. InstantID needs a clear, "
                "front-facing face crop as the character reference."
            )
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))[-1]

    # ---- pipeline ------------------------------------------------------
    def _load_pipeline(
        self,
        settings: WorkerSettings,
        request: ImageRequest,
        model_target: dict[str, Any],
        progress: ProgressCallback,
        *,
        job_id: str,
    ) -> Any:
        torch = importlib.import_module("torch")
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))

        if self._pipe is not None and self._loaded_repo == repo:
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            self._loaded_model = request.model
            return self._pipe
        if self._pipe is not None:
            self.unload()

        instant = model_target.get("instantId") or {}
        if not instant:
            raise RuntimeError(f"{request.model} has no InstantID configuration.")
        instant_repo = instant.get("repo", "InstantX/InstantID")
        cache_action = "Loading cached" if huggingface_repo_cache_exists(repo) else "Downloading"
        progress("loading_model", "loading_model", 0.2, f"{cache_action} {model_target['label']} (InstantID).")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useInstantId=True,
        )

        pipeline_class, _ = _import_instantid()
        diffusers = importlib.import_module("diffusers")
        from huggingface_hub import hf_hub_download

        controlnet = diffusers.ControlNetModel.from_pretrained(
            instant_repo, subfolder=instant.get("controlnetSubfolder", "ControlNetModel"), torch_dtype=dtype
        )
        from_pretrained_kwargs: dict[str, Any] = {"controlnet": controlnet, "torch_dtype": dtype}
        if model_target.get("variant"):
            from_pretrained_kwargs["variant"] = model_target["variant"]
        try:
            pipe = pipeline_class.from_pretrained(repo, **from_pretrained_kwargs)
        except Exception:
            from_pretrained_kwargs.pop("variant", None)
            pipe = pipeline_class.from_pretrained(repo, **from_pretrained_kwargs)

        ip_bin = hf_hub_download(repo_id=instant_repo, filename=instant.get("ipAdapter", "ip-adapter.bin"))
        pipe.load_ip_adapter_instantid(ip_bin)
        pipe.to(device)
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

        emit_worker_event("image_pipeline_load_complete", jobId=job_id, adapter=self.id, model=request.model, repo=repo)
        self._pipe = pipe
        self._loaded_repo = repo
        self._loaded_model = request.model
        return pipe

    def _ip_adapter_scale(self, request: ImageRequest) -> float:
        try:
            return float(request.advanced.get("ipAdapterScale", 0.8))
        except (TypeError, ValueError):
            return 0.8

    def _controlnet_scale(self, request: ImageRequest) -> float:
        try:
            return float(request.advanced.get("controlnetConditioningScale", 0.8))
        except (TypeError, ValueError):
            return 0.8

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target.get("steps", 30), 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = model_target.get("guidanceScale", 5.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _run_pipeline(
        self,
        settings: WorkerSettings,
        pipe: Any,
        request: ImageRequest,
        seed: int,
        project_path: Path,
        cancel_requested: CancelCallback | None = None,
    ) -> Image.Image:
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        _, draw_kps = _import_instantid()
        model_target = MODEL_TARGETS[request.model]

        reference = load_reference_image(project_path, request.reference_asset_id)
        canvas = _letterbox(reference, request.width, request.height)
        face = self._largest_face(canvas)
        face_emb = face["embedding"]
        face_kps = draw_kps(canvas, face["kps"])

        pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "image_embeds": face_emb,
            "image": face_kps,
            "controlnet_conditioning_scale": self._controlnet_scale(request),
            "ip_adapter_scale": self._ip_adapter_scale(request),
            "width": request.width,
            "height": request.height,
            "num_inference_steps": self._num_inference_steps(request, model_target),
            "guidance_scale": self._guidance_scale(request, model_target),
            "generator": generator,
        }
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**filter_call_kwargs(pipe, kwargs))
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        return output.images[0].convert("RGB")

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
            raise RuntimeError(f"{request.model} is not an InstantID target.")
        if request.mode == "edit_image":
            raise RuntimeError(f"{request.model} does not support image editing.")
        if not request.reference_asset_id:
            raise RuntimeError("InstantID generation requires a character reference image.")
        _require_instantid_extras()

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']} (InstantID).")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = model_target["label"]

        def image_at_index(index: int) -> Image.Image:
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            progress(
                "running",
                "generating",
                image_batch_progress(index, request.count),
                format_batch_running_message(label, index, request.count),
            )
            emit_worker_event(
                "image_inference_start",
                jobId=job["id"],
                adapter=self.id,
                model=request.model,
                imageIndex=index,
                imageCount=request.count,
                device=device,
            )
            try:
                image = self._run_pipeline(
                    settings, pipe, request, seed, project_path, cancel_requested=cancel_requested
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
            image_count=request.count,
            image_at_index=image_at_index,
            adapter_id=self.id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings={
                **request.advanced,
                "repo": request.advanced.get("modelRepo") or model_target["repo"],
                "instantId": True,
                "ipAdapterScale": self._ip_adapter_scale(request),
                "controlnetConditioningScale": self._controlnet_scale(request),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request, model_target),
                "realModelInference": True,
            },
            settings=settings,
            job_id=job["id"],
        )
