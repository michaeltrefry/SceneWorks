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
from PIL import Image, ImageDraw, ImageFilter

from .openpose_skeleton import draw_bodypose, face_box_from_keypoints, normalize_keypoints
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
from .lora_adapters import LoraPipelineState, apply_loras_to_pipeline
from .sampler_registry import apply_sampler, sampler_selection_from_advanced
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


# Canonical view-angle landmark sets (insightface 5-point kps, order
# [left_eye, right_eye, nose, mouth_left, mouth_right]), normalized to a SQUARE
# canvas. sc-2009: extracted from a 9-cell turnaround sheet + full L/R profiles,
# each validated to hold InstantID identity (cosine 0.81-0.89) at the target angle.
# The pack supplies the IdentityNet *pose* while the character reference supplies
# *identity*, so one constant rotates any character to any of these views. View-angle
# generation outputs square (the kps must share the output aspect — the sc-2009
# kps-distortion rule), and the reference image is still used for the face embedding.
VIEW_ANGLE_KPS: dict[str, list[tuple[float, float]]] = {
    "front": [(0.4460, 0.5227), (0.5755, 0.5166), (0.5106, 0.5947), (0.4653, 0.6660), (0.5630, 0.6613)],
    "three_quarter_left": [(0.3679, 0.5325), (0.4514, 0.5354), (0.3553, 0.6007), (0.3724, 0.6718), (0.4349, 0.6733)],
    "three_quarter_right": [(0.5946, 0.4930), (0.6882, 0.4955), (0.6948, 0.5598), (0.6202, 0.6408), (0.6885, 0.6421)],
    "left_profile": [(0.4373, 0.3527), (0.4925, 0.3445), (0.3927, 0.4662), (0.4853, 0.5599), (0.5240, 0.5517)],
    "right_profile": [(0.5075, 0.3445), (0.5627, 0.3527), (0.6073, 0.4662), (0.4760, 0.5517), (0.5147, 0.5599)],
    "up": [(0.4535, 0.4371), (0.5765, 0.4332), (0.5077, 0.4918), (0.4647, 0.5704), (0.5646, 0.5667)],
    "down": [(0.4457, 0.6231), (0.5848, 0.6228), (0.5174, 0.7337), (0.4726, 0.7771), (0.5645, 0.7770)],
    "up_left": [(0.3757, 0.4584), (0.4504, 0.4681), (0.3490, 0.4918), (0.3430, 0.5857), (0.3936, 0.5924)],
    "up_right": [(0.5787, 0.4431), (0.6673, 0.4337), (0.6799, 0.4601), (0.6331, 0.5601), (0.6989, 0.5515)],
    "down_left": [(0.3344, 0.6464), (0.4363, 0.6282), (0.3749, 0.7418), (0.4090, 0.7905), (0.4662, 0.7762)],
    "down_right": [(0.5963, 0.6165), (0.6823, 0.6271), (0.6650, 0.7171), (0.5668, 0.7524), (0.6198, 0.7640)],
}


def _view_angle_kps(angle: str, side: int) -> np.ndarray | None:
    """Scaled (side x side) landmark array for a named view angle, or None if unknown."""
    points = VIEW_ANGLE_KPS.get(angle)
    if not points:
        return None
    return np.array(points, dtype=np.float32) * float(side)


# Order the one-click "angle set" (advanced.angleSet) generates the character in:
# front, the two three-quarters, full profiles, then the up/down/diagonal tilts.
ANGLE_SET_ORDER: tuple[str, ...] = (
    "front", "three_quarter_left", "three_quarter_right", "left_profile", "right_profile",
    "up", "down", "up_left", "up_right", "down_left", "down_right",
)

# Pose-library generation (advanced.poses) renders on a SQUARE canvas (the library
# skeletons are square, and the OpenPose control image must share the output aspect —
# the kps-distortion rule). The face is small at full-body framing, so a face-restoration
# pass re-imposes identity afterward (sc-2063 spike: ~0.38 -> ~0.88 ArcFace cosine).
_POSE_SIZE: int = 1024
# Gender-neutral on purpose (sc-3381): this prompt is applied to EVERY character during the
# face-restore re-render, so it must not assume a gender. (The native MLX engine's
# `FACE_RESTORE_PROMPT` is the same neutral wording.)
_FACE_RESTORE_PROMPT = "close-up portrait of the face, soft natural light, photorealistic, sharp focus"


class InstantIDAdapter:
    """Identity-preserving SDXL generation via InstantID (face embedding + IdentityNet)."""

    id = "instantid_sdxl"

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_model: str | None = None
        # "identity" (IdentityNet only) vs "multi" (IdentityNet + OpenPose). A job that
        # switches between view-angle and full-body-pose modes reloads the pipeline.
        self._loaded_controlnet_mode: str | None = None
        # SDXL LoRA merge state for the single cached pipe (sc-2224). One field rather
        # than the dict-keyed cache other adapters use, because this adapter holds at
        # most one pipe at a time and reloads (clearing this) on repo/mode change.
        self._loaded_lora_state = LoraPipelineState()
        self._face_app: Any | None = None

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._loaded_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        if self._pipe is None:
            return False
        self._pipe = None
        self._loaded_repo = None
        self._loaded_model = None
        self._loaded_controlnet_mode = None
        self._loaded_lora_state = LoraPipelineState()
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
        pose_set: bool = False,
    ) -> Any:
        torch = importlib.import_module("torch")
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        # Full-body pose set adds an OpenPose ControlNet (MultiControlNet); other modes
        # use IdentityNet alone. The mode is part of the cache key so a job that changes
        # mode reloads rather than feeding the wrong number of control images.
        mode = "multi" if pose_set else "identity"

        if self._pipe is not None and self._loaded_repo == repo and self._loaded_controlnet_mode == mode:
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
            controlnetMode=mode,
        )

        pipeline_class, _ = _import_instantid()
        diffusers = importlib.import_module("diffusers")
        from huggingface_hub import hf_hub_download

        identitynet = diffusers.ControlNetModel.from_pretrained(
            instant_repo, subfolder=instant.get("controlnetSubfolder", "ControlNetModel"), torch_dtype=dtype
        )
        if pose_set:
            open_pose = model_target.get("openPose") or {}
            open_pose_repo = open_pose.get("repo")
            if not open_pose_repo:
                raise RuntimeError(f"{request.model} has no OpenPose configuration for the full-body pose set.")
            openpose_net = diffusers.ControlNetModel.from_pretrained(open_pose_repo, torch_dtype=dtype)
            try:  # diffusers >=0.34 canonical path; old re-export errors on instantiation
                from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
            except ImportError:
                from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
            controlnet: Any = MultiControlNetModel([identitynet, openpose_net])
        else:
            controlnet = identitynet
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
        self._loaded_controlnet_mode = mode
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

    def _openpose_scale(self, request: ImageRequest) -> float:
        try:
            return float(request.advanced.get("openPoseScale", 0.7))
        except (TypeError, ValueError):
            return 0.7

    @staticmethod
    def _face_restore_enabled(request: ImageRequest) -> bool:
        """Whether the full-body face-restoration pass runs (advanced.faceRestore,
        default off). Off = the OpenPose+InstantID base image is used as-is (cleaner
        blend, but weaker identity at the small full-body face size)."""
        value = request.advanced.get("faceRestore", False)
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    @staticmethod
    def _normalized_kps(face: Any) -> np.ndarray:
        """Reference 5-point kps normalized to the face bbox (so they can be re-placed at
        any position/scale on a new canvas)."""
        kps = np.asarray(face["kps"], dtype=np.float32)
        x1, y1, x2, y2 = face["bbox"]
        origin = np.array([x1, y1], dtype=np.float32)
        size = np.array([max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32)
        return (kps - origin) / size

    def _run_pose(
        self,
        settings: WorkerSettings,
        pipe: Any,
        request: ImageRequest,
        seed: int,
        project_path: Path,
        keypoints: list[Any],
        cancel_requested: CancelCallback | None = None,
    ) -> Image.Image:
        """Generate the character in one library pose (square canvas): the OpenPose
        skeleton (rendered from `keypoints`) drives the pose, IdentityNet anchors the face
        when the head is visible, then the face-restoration pass re-imposes identity at the
        small full-body face size."""
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        _, draw_kps = _import_instantid()
        model_target = MODEL_TARGETS[request.model]
        width = height = _POSE_SIZE

        reference = load_reference_image(project_path, request.reference_asset_id)
        face = self._largest_face(_letterbox(reference, width, height))
        face_emb = face["embedding"]
        skeleton = Image.fromarray(draw_bodypose(width, height, keypoints))
        face_box = face_box_from_keypoints(keypoints)
        openpose_scale = self._openpose_scale(request)

        if face_box is not None:
            cx, cy, face_h_frac = face_box
            norm = self._normalized_kps(face)
            x1, y1, x2, y2 = face["bbox"]
            aspect = max(1.0, x2 - x1) / max(1.0, y2 - y1)
            face_h = height * face_h_frac
            face_w = face_h * aspect
            placed = norm.copy()
            placed[:, 0] = cx * width + (placed[:, 0] - 0.5) * face_w
            placed[:, 1] = cy * height + (placed[:, 1] - 0.5) * face_h
            face_kps = draw_kps(Image.new("RGB", (width, height), (0, 0, 0)), placed)
            control_images = [face_kps, skeleton]
            control_scales = [self._controlnet_scale(request), openpose_scale]
            ip_scale = self._ip_adapter_scale(request)
        else:
            # No visible face (e.g. a back view or occluded head): OpenPose only; the
            # shared seed + prompt carry hair/wardrobe continuity. Disable IdentityNet +
            # ip-adapter so no face is forced onto the back of the head.
            control_images = [Image.new("RGB", (width, height), (0, 0, 0)), skeleton]
            control_scales = [0.0, max(openpose_scale, 0.85)]
            ip_scale = 0.0

        prompt = request.prompt
        pipe.set_ip_adapter_scale(ip_scale)
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": request.negative_prompt,
            "image_embeds": face_emb,
            "image": control_images,
            "controlnet_conditioning_scale": control_scales,
            "ip_adapter_scale": ip_scale,
            "width": width,
            "height": height,
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
        base = output.images[0].convert("RGB")
        if face_box is not None and self._face_restore_enabled(request):
            base = self._restore_face(settings, pipe, request, base, face_emb, seed, cancel_requested)
        return base

    def _restore_face(
        self,
        settings: WorkerSettings,
        pipe: Any,
        request: ImageRequest,
        base: Image.Image,
        reference_embedding: Any,
        seed: int,
        cancel_requested: CancelCallback | None = None,
    ) -> Image.Image:
        """ADetailer-style identity restoration: detect the (small) face in a full-body
        image, crop + upscale to 1024, re-run InstantID on the crop with the reference
        embedding, and paste it back with a feathered mask. Recovers ArcFace identity
        from ~0.38 to ~0.88 at full-body framing (sc-2063). No-op if no face is found."""
        import cv2

        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        _, draw_kps = _import_instantid()
        model_target = MODEL_TARGETS[request.model]

        bgr = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
        faces = self._face_analysis().get(bgr)
        if not faces:
            return base
        face = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))[-1]
        x1, y1, x2, y2 = face["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * 1.9 / 2
        a, b = int(max(0, cx - half)), int(max(0, cy - half))
        c, d = int(min(base.width, cx + half)), int(min(base.height, cy + half))
        crop_w, crop_h = c - a, d - b
        if crop_w < 16 or crop_h < 16:
            return base

        side = 1024
        kps = np.asarray(face["kps"], dtype=np.float32).copy()
        kps[:, 0] = (kps[:, 0] - a) / crop_w * side
        kps[:, 1] = (kps[:, 1] - b) / crop_h * side
        crop_kps = draw_kps(Image.new("RGB", (side, side), (0, 0, 0)), kps)
        blank = Image.new("RGB", (side, side), (0, 0, 0))

        pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": _FACE_RESTORE_PROMPT,
            "negative_prompt": request.negative_prompt,
            "image_embeds": reference_embedding,
            # Only IdentityNet acts on the crop; OpenPose is zeroed out.
            "image": [crop_kps, blank],
            "controlnet_conditioning_scale": [self._controlnet_scale(request), 0.0],
            "ip_adapter_scale": self._ip_adapter_scale(request),
            "width": side,
            "height": side,
            "num_inference_steps": self._num_inference_steps(request, model_target),
            "guidance_scale": self._guidance_scale(request, model_target),
            "generator": generator,
        }
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        restored = pipe(**filter_call_kwargs(pipe, kwargs)).images[0].convert("RGB")
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")

        small = restored.resize((crop_w, crop_h), Image.LANCZOS)
        mask = Image.new("L", (crop_w, crop_h), 0)
        ImageDraw.Draw(mask).ellipse(
            [int(crop_w * 0.1), int(crop_h * 0.1), int(crop_w * 0.9), int(crop_h * 0.9)], fill=255
        )
        mask = mask.filter(ImageFilter.GaussianBlur(max(4, crop_w // 12)))
        composited = base.copy()
        composited.paste(small, (a, b), mask)
        return composited

    @staticmethod
    def _view_angle(request: ImageRequest) -> str | None:
        """The requested canonical view angle (advanced.viewAngle), or None to take the
        pose from the reference image's own landmarks (default behavior)."""
        angle = request.advanced.get("viewAngle")
        if isinstance(angle, str) and angle in VIEW_ANGLE_KPS:
            return angle
        return None

    def _run_pipeline(
        self,
        settings: WorkerSettings,
        pipe: Any,
        request: ImageRequest,
        seed: int,
        project_path: Path,
        cancel_requested: CancelCallback | None = None,
        view_angle_override: str | None = None,
    ) -> Image.Image:
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        _, draw_kps = _import_instantid()
        model_target = MODEL_TARGETS[request.model]

        reference = load_reference_image(project_path, request.reference_asset_id)
        # An explicit per-call angle (the angle-set batch) wins over advanced.viewAngle.
        view_angle = view_angle_override if view_angle_override in VIEW_ANGLE_KPS else self._view_angle(request)
        if view_angle is not None:
            # View-angle mode: pose from the canonical landmark pack, identity from the
            # reference embedding. Square canvas so the pack kps share the output aspect
            # (kps-distortion rule); the requested dims only set the square side.
            side = max(256, min(request.width, request.height))
            side -= side % 8
            face = self._largest_face(_letterbox(reference, side, side))
            face_kps = draw_kps(Image.new("RGB", (side, side), (0, 0, 0)), _view_angle_kps(view_angle, side))
            width = height = side
        else:
            canvas = _letterbox(reference, request.width, request.height)
            face = self._largest_face(canvas)
            face_kps = draw_kps(canvas, face["kps"])
            width, height = request.width, request.height
        face_emb = face["embedding"]

        pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "image_embeds": face_emb,
            "image": face_kps,
            "controlnet_conditioning_scale": self._controlnet_scale(request),
            "ip_adapter_scale": self._ip_adapter_scale(request),
            "width": width,
            "height": height,
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

        # Pose library (advanced.poses): a list of {id, keypoints} selected from the pose
        # gallery — generate the character once per pose in a single job, each with a
        # face-restoration pass. Needs the MultiControlNet (IdentityNet + OpenPose) pipe.
        raw_poses = request.advanced.get("poses")
        pose_entries = [p for p in raw_poses if isinstance(p, dict)] if isinstance(raw_poses, list) else []
        pose_set = len(pose_entries) > 0
        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']} (InstantID).")
        pipe = self._load_pipeline(
            settings, request, model_target, progress=progress, job_id=job["id"], pose_set=pose_set
        )
        # Apply any selected SDXL LoRAs to the (cached) pipe once, before the angle/pose
        # loop — every image in the set reuses this pipe, and the merge persists across
        # the per-pose _restore_face pass (validated by the sc-2222 spike). previous_state
        # tracks the single live pipe (reset on reload via unload());
        # validate_lora_compatibility (family "sdxl") rejects incompatible LoRAs.
        self._loaded_lora_state = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            model_id=request.model,
            previous_state=self._loaded_lora_state,
        )
        # Optional sampler / scheduler swap (sc-3857). InstantID is an SDXL
        # (epsilon) pipe, so the registry routes it to the standard solver table
        # — e.g. dpmpp_sde + karras = "DPM++ SDE Karras" (the RealVisXL-
        # recommended combo, sharper than the default discrete schedule). A
        # "default"/"default" selection is a true no-op; applied once on the
        # cached pipe before the angle/pose loop so every image in the set shares
        # the same scheduler.
        sampler_key, scheduler_key, scheduler_shift = sampler_selection_from_advanced(request.advanced)
        apply_sampler(pipe, sampler_key, scheduler_key, scheduler_shift, adapter=self.id)
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = model_target["label"]
        # One-click angle set (advanced.angleSet): generate the character once per packed
        # view angle in a single job (pipeline already loaded), instead of `count` copies
        # of one angle. Identity comes from the same reference embedding throughout.
        angle_set = bool(request.advanced.get("angleSet")) and not pose_set
        angles = [a for a in ANGLE_SET_ORDER if a in VIEW_ANGLE_KPS] if angle_set else []
        pose_keypoints = [normalize_keypoints(p.get("keypoints")) for p in pose_entries] if pose_set else []
        if pose_set:
            total = len(pose_entries)
        elif angle_set:
            total = len(angles)
        else:
            total = request.count
        # Every image in a set shares ONE seed so the noise-derived attributes InstantID
        # does NOT lock (hair, wardrobe, lighting) stay consistent across the set — only
        # the pose changes. Plain batches keep per-image seeds for variety.
        set_seed = resolve_seed(request.seed, request.prompt, 0, request.seeds)

        def image_at_index(index: int) -> Image.Image:
            grouped = angle_set or pose_set
            seed = set_seed if grouped else resolve_seed(request.seed, request.prompt, index, request.seeds)
            angle = angles[index] if angle_set else None
            pose_id = pose_entries[index].get("id") if pose_set else None
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
                viewAngle=angle,
                poseId=pose_id,
                device=device,
            )
            try:
                if pose_set:
                    image = self._run_pose(
                        settings, pipe, request, seed, project_path, pose_keypoints[index],
                        cancel_requested=cancel_requested,
                    )
                else:
                    image = self._run_pipeline(
                        settings,
                        pipe,
                        request,
                        seed,
                        project_path,
                        cancel_requested=cancel_requested,
                        view_angle_override=angle,
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
                "instantId": True,
                "ipAdapterScale": self._ip_adapter_scale(request),
                "controlnetConditioningScale": self._controlnet_scale(request),
                **(
                    {
                        "openPoseScale": self._openpose_scale(request),
                        "poseLibrary": True,
                        "faceRestore": self._face_restore_enabled(request),
                    }
                    if pose_set
                    else {}
                ),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request, model_target),
                "realModelInference": True,
            },
            settings=settings,
            job_id=job["id"],
        )
