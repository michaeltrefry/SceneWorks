from __future__ import annotations

from dataclasses import dataclass, field
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol, TypeVar
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from sceneworks_shared import (
    find_asset_sidecar_path,
    find_project_path as shared_find_project_path,
    index_asset,
    read_json,
    safe_int,
    slugify,
    utc_now,
    write_json,
)

from .adapter_utils import cancel_step_callback, filter_call_kwargs
from .character_studio_angles import (
    ANGLE_PROMPT_AUGMENTS,
    ANGLE_SET_ORDER as CHARACTER_ANGLE_SET_ORDER,
    augment_prompt_for_angle,
    augment_prompt_for_pose,
)
from .openpose_skeleton import draw_bodypose, draw_wholebody, normalize_face, normalize_hands, normalize_keypoints
from .sampler_registry import apply_sampler, sampler_selection_from_advanced
from .hf_cache import huggingface_cache_roots, huggingface_repo_cache_path
from .lora_adapters import (
    LoraPipelineState,
    adapter_network_type,
    classify_adapter_network,
    apply_loras_to_pipeline,
    lora_path,
    normalize_lora_specs,
    reject_loras_if_unsupported,
    reject_lokr_loras,
    validate_lora_compatibility,
)
from .settings import WorkerSettings
from .upscalers import RealESRGANUpscaler, UpscaleJob

if TYPE_CHECKING:
    # instantid_adapter and pulid_flux_adapter both import from this module, so
    # they can only be referenced for typing (their runtime instances are created
    # in runtime.py and passed in via the adapters dict; create_image_adapter
    # lazily imports each on the no-dict path).
    from .instantid_adapter import InstantIDAdapter
    from .pulid_flux_adapter import PuLIDFluxAdapter


Image.MAX_IMAGE_PIXELS = 64_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

CancelCallback = Callable[[], bool]


class ProgressCallback(Protocol):
    def __call__(
        self,
        status: str,
        stage: str,
        value: float,
        message: str,
        result: dict[str, Any] | None = None,
    ) -> None: ...


def huggingface_repo_cache_exists(repo: str) -> bool:
    repo_cache = huggingface_repo_cache_path(repo)
    if repo_cache is None:
        return False
    return (repo_cache / "snapshots").is_dir() or (repo_cache / "blobs").is_dir()


def emit_worker_event(event: str, **payload: Any) -> None:
    """Emit a structured JSON diagnostic event on the worker's stdout.

    Mirrors `scene_worker.runtime.emit` so adapter-level phase markers
    (pipeline load, device placement, per-image inference) land in the
    same operator log stream as worker lifecycle events. Keeps phases
    distinguishable when a generation job appears to hang.
    """

    payload["event"] = event
    payload["reportedAt"] = utc_now()
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def gpu_memory_snapshot(torch: Any, device: str) -> dict[str, Any] | None:
    if isinstance(device, str) and device.startswith("mps"):
        # MPS uses unified memory, so allocated/driver figures double as the
        # process's accelerator footprint — the only built-in signal for the Mac
        # memory growth (CUDA-style per-device stats don't exist here).
        mps_backend = getattr(torch, "mps", None)
        if mps_backend is None:
            return None
        snapshot: dict[str, Any] = {"device": "mps"}
        for key, attr in (
            ("allocatedMb", "current_allocated_memory"),
            ("driverAllocatedMb", "driver_allocated_memory"),
        ):
            fn = getattr(mps_backend, attr, None)
            if callable(fn):
                try:
                    snapshot[key] = round(int(fn()) / (1024 * 1024), 2)
                except Exception:
                    pass
        return snapshot
    if not isinstance(device, str) or not device.startswith("cuda"):
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return None
    try:
        if not bool(cuda.is_available()):
            return None
    except Exception:
        return None
    snapshot: dict[str, Any] = {"device": device}
    index = None
    if ":" in device:
        try:
            index = int(device.split(":", 1)[1])
        except ValueError:
            index = None
    try:
        allocated = int(cuda.memory_allocated(index) if index is not None else cuda.memory_allocated())
        snapshot["allocatedMb"] = round(allocated / (1024 * 1024), 2)
    except Exception:
        pass
    try:
        reserved = int(cuda.memory_reserved(index) if index is not None else cuda.memory_reserved())
        snapshot["reservedMb"] = round(reserved / (1024 * 1024), 2)
    except Exception:
        pass
    return snapshot


def pipeline_component_devices(pipe: Any) -> list[str]:
    """Return the sorted, unique torch device strings of a pipeline's submodules."""

    devices: set[str] = set()
    components = getattr(pipe, "components", None)
    if isinstance(components, dict):
        candidates = list(components.values())
    else:
        candidate_names = ("transformer", "unet", "text_encoder", "text_encoder_2", "vae")
        candidates = [getattr(pipe, name, None) for name in candidate_names]
    for component in candidates:
        if component is None:
            continue
        device = getattr(component, "device", None)
        if device is None:
            parameters = getattr(component, "parameters", None)
            if callable(parameters):
                try:
                    first = next(parameters())
                except StopIteration:
                    first = None
                except Exception:
                    first = None
                if first is not None:
                    device = getattr(first, "device", None)
        if device is None:
            continue
        devices.add(str(device))
    return sorted(devices)


def verify_pipeline_on_device(
    pipe: Any,
    *,
    requested_device: str,
    model_label: str,
    allow_offload: bool,
) -> list[str]:
    """Confirm a GPU-bound pipeline actually landed on the requested CUDA device.

    Returns the observed component device strings. Raises RuntimeError when
    the worker asked for a CUDA device but no pipeline component is on a
    matching CUDA device — that path is the most common source of jobs that
    look "running" while the GPU stays idle.
    """

    devices = pipeline_component_devices(pipe)
    if allow_offload or not requested_device.startswith("cuda"):
        return devices
    if not devices:
        return devices
    target_index = requested_device.split(":", 1)[1] if ":" in requested_device else None
    unexpected_devices = []
    for device in devices:
        if target_index is None:
            if device == "cuda" or device.startswith("cuda:"):
                continue
        elif device == "cuda" or device == requested_device:
            continue
        unexpected_devices.append(device)
    if not unexpected_devices:
        return devices
    observed = ", ".join(devices) or "no detected device"
    raise RuntimeError(
        f"{model_label} did not move onto {requested_device}; pipeline components are on {observed}. "
        "Check CUDA driver compatibility and worker GPU assignment, then retry."
    )


def format_batch_running_message(label: str, index: int, total: int) -> str:
    """Build a per-iteration "Running" progress message that names the actual
    saved count alongside the in-flight index, so users do not see "Running 3
    of 4" without prior images being durable."""

    prefix = f"Generated {index} of {total}. " if index > 0 else ""
    return f"{prefix}Running {label} {index + 1} of {total}."


MODEL_TARGETS = {
    "z_image_turbo": {
        "label": "Z-Image-Turbo",
        "family": "z-image",
        "supportsEdit": False,
        "steps": 8,
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "adapter": "z_image_diffusers",
    },
    "z_image_edit": {
        "label": "Z-Image-Edit",
        "family": "z-image",
        "supportsEdit": True,
        "steps": 8,
        # Uses Turbo weights via ZImageImg2ImgPipeline until the dedicated Edit checkpoint is released.
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "adapter": "z_image_diffusers",
    },
    "qwen_image": {
        "label": "Qwen Image",
        "family": "qwen-image",
        "supportsEdit": False,
        "steps": 20,
        "repo": "Qwen/Qwen-Image",
        "adapter": "qwen_image",
    },
    # sc-2160: qwen_image_edit (Aug 2025) and qwen_image_edit_2509 (Sep 2025) are
    # aliased to Qwen-Image-Edit-2511 (Dec 2025) — the 2511 weights are a drop-in
    # successor on the same QwenImageEditPlusPipeline, with drift mitigation,
    # multi-person consistency, integrated popular LoRAs, and stronger geometric
    # reasoning. Keeping the legacy IDs here means old jobs/presets/characters
    # pinned to them still resolve. The manifest exposes only qwen_image_edit_2511
    # + qwen_image_edit_2511_lightning in the picker.
    "qwen_image_edit": {
        "label": "Qwen Image Edit",
        "family": "qwen-image",
        "supportsEdit": True,
        "steps": 40,
        "guidanceScale": 1.0,
        "repo": "Qwen/Qwen-Image-Edit-2511",
        "adapter": "qwen_image",
    },
    "qwen_image_edit_2509": {
        "label": "Qwen Image Edit (2509)",
        "family": "qwen-image",
        "supportsEdit": True,
        "steps": 40,
        "guidanceScale": 1.0,
        "repo": "Qwen/Qwen-Image-Edit-2511",
        "adapter": "qwen_image",
    },
    "qwen_image_edit_2511": {
        "label": "Qwen Image Edit (2511)",
        "family": "qwen-image",
        "supportsEdit": True,
        # Model-card defaults: 40 steps + true_cfg_scale 4.0 + guidance_scale 1.0.
        # Same QwenImageEditPlusPipeline as the 2509 release (multi-image-capable);
        # 2511 improvements are in the weights, not the API. Apache-2.0, ungated.
        "steps": 40,
        "guidanceScale": 1.0,
        "repo": "Qwen/Qwen-Image-Edit-2511",
        "adapter": "qwen_image",
    },
    "qwen_image_edit_2511_lightning": {
        "label": "Qwen Image Edit (2511) Lightning",
        "family": "qwen-image",
        "supportsEdit": True,
        # 4-step distill: cfg 1.0 / true_cfg_scale 1.0 / 4 steps (vs base 40).
        # lightx2v Lightning LoRA fuses into the 2511 base on load; user LoRAs
        # still stack on top via the normal apply_loras_to_pipeline path.
        "steps": 4,
        "guidanceScale": 1.0,
        "trueCfgScale": 1.0,
        "repo": "Qwen/Qwen-Image-Edit-2511",
        "adapter": "qwen_image",
        "distillLora": {
            "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        },
    },
    "lens": {
        "label": "Lens",
        "family": "lens",
        "supportsEdit": False,
        # Non-distilled base: 20 steps, CFG 5.0. Also the LoRA training base.
        "steps": 20,
        "guidanceScale": 5.0,
        "repo": "microsoft/Lens",
        "adapter": "lens_turbo",
    },
    "lens_turbo": {
        "label": "Lens-Turbo",
        "family": "lens",
        "supportsEdit": False,
        # Distilled 4-step variant; the base Lens model uses 20-50 steps.
        "steps": 4,
        "repo": "microsoft/Lens-Turbo",
        "adapter": "lens_turbo",
    },
    "sensenova_u1_8b": {
        "label": "SenseNova-U1 8B",
        "family": "sensenova-u1",
        # Unified model: same weights do text-to-image and instruction editing (it2i).
        "supportsEdit": True,
        # Base 8B-MoT uses ~50 steps; an 8-step distill LoRA exists (cfg 1.0).
        "steps": 50,
        "repo": "sensenova/SenseNova-U1-8B-MoT",
        "adapter": "sensenova_u1",
    },
    "sensenova_u1_8b_fast": {
        "label": "SenseNova-U1 8B Fast",
        "family": "sensenova-u1",
        # Distilled editing (it2i) at 8 steps; the it2i path merges the same LoRA.
        "supportsEdit": True,
        # 8-step distill LoRA (cfg 1.0): shares the base weights, ~5-6x faster.
        "steps": 8,
        "guidanceScale": 1.0,
        "repo": "sensenova/SenseNova-U1-8B-MoT",
        "adapter": "sensenova_u1",
        "distillLora": {
            "repo": "sensenova/SenseNova-U1-8B-MoT-LoRAs",
            "file": "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors",
        },
    },
    "flux_schnell": {
        "label": "FLUX.1 [schnell]",
        "family": "flux",
        "supportsEdit": False,
        # Guidance-distilled: ignores CFG (guidance 0), ~4 steps; T5 max_seq_len 256.
        "steps": 4,
        "guidanceScale": 0.0,
        "maxSequenceLength": 256,
        "repo": "black-forest-labs/FLUX.1-schnell",
        "adapter": "flux_diffusers",
    },
    "flux_dev": {
        "label": "FLUX.1 [dev]",
        "family": "flux",
        "supportsEdit": False,
        # Guided: guidance ~3.5, ~28 steps; T5 max_seq_len 512. Gated (non-commercial).
        "steps": 28,
        "guidanceScale": 3.5,
        "maxSequenceLength": 512,
        "repo": "black-forest-labs/FLUX.1-dev",
        "adapter": "flux_diffusers",
        # XLabs FLUX IP-Adapter for Character Studio reference (sc-2011). The
        # diffusers-blessed path: FluxIPAdapterMixin natively handles encoder load,
        # weight load, and scale setting; FluxPipeline takes ip_adapter_image=
        # alongside true_cfg_scale for real CFG against the negative prompt (FLUX
        # is otherwise guidance-distilled). License: FLUX.1 [dev] NC — same posture
        # as the base flux_dev built-in (already NC + gated). schnell has no native
        # IP-Adapter, so this block lives only on flux_dev.
        "ipAdapter": {
            "repo": "XLabs-AI/flux-ip-adapter",
            "weight": "ip_adapter.safetensors",
            "imageEncoderRepo": "openai/clip-vit-large-patch14",
        },
    },
    "flux2_klein_9b": {
        "label": "FLUX.2 [klein] 9B",
        "family": "flux2-klein",
        # Unified model: txt2img via Flux2Klein, edit via Flux2KleinEdit (reference
        # image conditioning). Both paths go through the same MLX-only adapter.
        "supportsEdit": True,
        # 4-step distilled (mflux's default for klein-9b); guidance 1.0 mandatory
        # on distilled variants — the mflux CLI errors on any other value.
        "steps": 4,
        "guidanceScale": 1.0,
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "adapter": "mlx_flux2",
    },
    "flux2_klein_9b_kv": {
        "label": "FLUX.2 [klein] 9B-KV",
        "family": "flux2-klein",
        # Full txt2img + edit, same as the base 9B (sc-2173 validated the
        # -kv distill renders plain text-to-image + non-character edit with
        # no artifacts). The KV cache auto-engages only on the edit path when
        # a reference is present, via the supports_kv_cache ModelConfig flag
        # in the pinned mflux fork (sc-2163, upstream PR filipstrand/mflux#426).
        "supportsEdit": True,
        "supportsTxt2Img": True,
        "steps": 4,
        "guidanceScale": 1.0,
        "repo": "black-forest-labs/FLUX.2-klein-9b-kv",
        "adapter": "mlx_flux2",
    },
    "flux2_klein_9b_true_v2": {
        "label": "FLUX.2 [klein] 9B True V2",
        "family": "flux2-klein",
        # Community full fine-tune of FLUX.2-klein-9B (wikeeyang) — improved
        # realism + prompt adherence (sc-2220). Same 9B architecture; weights
        # ship as a transformer-only single-file checkpoint converted at install
        # into a local diffusers dir (sc-2235) and loaded via the runner's
        # modelPath seam. This is the UNDISTILLED line, so unlike the 4-step
        # distill it wants ~24 steps at guidance 1.0 (sc-2220 / V1 model card).
        "supportsEdit": True,
        "supportsTxt2Img": True,
        "steps": 24,
        "guidanceScale": 1.0,
        # The wikeeyang source repo; the install-time convert job pulls the bf16
        # single-file from here. The runtime weights load from the assembled
        # local dir (see MlxFlux2Adapter._local_model_dir), not this repo.
        "repo": "wikeeyang/Flux2-Klein-9B-True-V2",
        "adapter": "mlx_flux2",
        # Borrowed VAE/text-encoder/tokenizer come from this base klein install.
        "componentBaseRepo": "black-forest-labs/FLUX.2-klein-9B",
    },
    "kolors": {
        "label": "Kolors",
        "family": "kolors",
        # Unified checkpoint: KolorsPipeline (T2I) + KolorsImg2ImgPipeline (edit).
        "supportsEdit": True,
        # Real CFG (not distilled): model card recommends guidance ~5.0, ~25 steps
        # with DPMSolverMultistep (Karras). ChatGLM3 text encoder max_seq_len 256.
        "steps": 25,
        "guidanceScale": 5.0,
        "maxSequenceLength": 256,
        # Kolors-diffusers ships fp16-variant weights only (*.fp16.safetensors),
        # so from_pretrained must request the fp16 variant.
        "variant": "fp16",
        "repo": "Kwai-Kolors/Kolors-diffusers",
        "adapter": "kolors_diffusers",
        # IP-Adapter-Plus (general reference image) for Character Studio "many
        # images from one reference". Needs its own CLIP-ViT-L-336 image encoder
        # and the safetensors weights from a PR revision. >24GB with the adapter.
        "ipAdapter": {
            "repo": "Kwai-Kolors/Kolors-IP-Adapter-Plus",
            "weight": "ip_adapter_plus_general.safetensors",
            "revision": "refs/pr/4",
        },
        # Strict pose tier (sc-2264): official Kolors pose ControlNet (DWPose-trained).
        # Loaded via the vendored Kolors_ControlNetModel + ControlNet img2img pipeline
        # (_vendor/kolors) — composes with IP-Adapter-Plus identity in one call.
        "controlNetPose": {
            "repo": "Kwai-Kolors/Kolors-ControlNet-Pose",
        },
    },
    "chroma1_hd": {
        "label": "Chroma1-HD",
        "family": "chroma",
        "supportsEdit": False,
        # Real CFG with negative prompts (unlike guidance-distilled FLUX.1-schnell):
        # ~40 steps, guidance 3.0; T5-XXL max_seq_len 512 (no CLIP encoder).
        "steps": 40,
        "guidanceScale": 3.0,
        "maxSequenceLength": 512,
        "repo": "lodestones/Chroma1-HD",
        "adapter": "chroma_diffusers",
    },
    "chroma1_base": {
        "label": "Chroma1-Base",
        "family": "chroma",
        "supportsEdit": False,
        # Neutral finetuning foundation; real CFG + negative prompts, ~40 steps, guidance 3.0.
        "steps": 40,
        "guidanceScale": 3.0,
        "maxSequenceLength": 512,
        "repo": "lodestones/Chroma1-Base",
        "adapter": "chroma_diffusers",
    },
    "chroma1_flash": {
        "label": "Chroma1-Flash",
        "family": "chroma",
        "supportsEdit": False,
        # CFG baked: ~8 steps, guidance 1.0 (CFG off, negative prompt ignored).
        "steps": 8,
        "guidanceScale": 1.0,
        "maxSequenceLength": 512,
        "repo": "lodestones/Chroma1-Flash",
        "adapter": "chroma_diffusers",
    },
    "sdxl": {
        "label": "Stable Diffusion XL",
        "family": "sdxl",
        # Unified base checkpoint: StableDiffusionXLPipeline (T2I) +
        # StableDiffusionXLImg2ImgPipeline (edit).
        "supportsEdit": True,
        # Real CFG with negative prompt (not distilled): ~30 steps at guidance
        # 7.0, native 1024x1024. Two CLIP text encoders, so no max_seq_len knob.
        # Ships EulerDiscreteScheduler, which we keep (scheduler UI is epic 1753).
        "steps": 30,
        "guidanceScale": 7.0,
        # SDXL base ships an fp16 variant alongside fp32; request it so
        # from_pretrained loads the half-precision weights.
        "variant": "fp16",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "adapter": "sdxl_diffusers",
        # IP-Adapter for SDXL (Character Studio reference). plus-face holds facial
        # structure via 16 patch tokens from a face-cropped training distribution,
        # so it carries more identity than non-plus (4 global tokens) without
        # copying clothing/composition as aggressively as plain plus (sc-2007).
        # ViT-H image encoder shipped in the same repo. Any SDXL-UNet model
        # (RealVisXL, etc.) can reuse this exact block. Stronger likeness still
        # belongs to InstantID (sc-2009); IP-Adapter is the resemblance tier.
        "ipAdapter": {
            "repo": "h94/IP-Adapter",
            "subfolder": "sdxl_models",
            "weight": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "encoderSubfolder": "models/image_encoder",
        },
    },
    "realvisxl": {
        "label": "RealVisXL (photoreal SDXL)",
        # SDXL UNet finetune — shares the sdxl LoRA family, sdxl_diffusers adapter,
        # and the same IP-Adapter block. Use this for plain photoreal SDXL t2i /
        # edit / reference work; InstantID (instantid_realvisxl) is the
        # face-identity engine on the same checkpoint (sc-2008 vs sc-2009).
        "family": "sdxl",
        "supportsEdit": True,
        # Real CFG with negative prompt: ~30 steps at guidance 7.0, native 1024.
        "steps": 30,
        "guidanceScale": 7.0,
        # RealVisXL_V5.0 ships fp16-variant weights.
        "variant": "fp16",
        # Photoreal SDXL finetune that solves the "shiny/plastic" complaint —
        # openrail++ (commercial use OK, ungated). Shares the HF cache with the
        # InstantID built-in below (single ~6.6 GiB download).
        "repo": "SG161222/RealVisXL_V5.0",
        "adapter": "sdxl_diffusers",
        # Same IP-Adapter as plain SDXL (h94/IP-Adapter is checkpoint-agnostic
        # within the SDXL UNet family); identity-faithful likeness still belongs
        # to instantid_realvisxl — IP-Adapter is the resemblance tier.
        "ipAdapter": {
            "repo": "h94/IP-Adapter",
            "subfolder": "sdxl_models",
            "weight": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "encoderSubfolder": "models/image_encoder",
        },
    },
    "instantid_realvisxl": {
        "label": "InstantID (RealVisXL)",
        # SDXL UNet under the hood, so it shares the sdxl LoRA family. Identity is
        # driven by an insightface ArcFace embedding + a 5-point-landmark ControlNet
        # ("IdentityNet"), NOT by an SDXL img2img/inpaint path — so it is strictly a
        # reference-driven character model (no plain text_to_image / edit_image).
        "family": "sdxl",
        "supportsEdit": False,
        # Per the sc-2009 spike: ~30 steps at guidance 5.0; identity holds with
        # ip_adapter_scale ~0.8 and controlnet_conditioning_scale 0.45 (looser pose)
        # .. 0.8 (frontal lock). Both ride advanced (ipAdapterScale /
        # controlnetConditioningScale); the adapter defaults them when absent.
        "steps": 30,
        "guidanceScale": 5.0,
        # RealVisXL ships fp16-variant weights; from_pretrained falls back to the
        # default precision if the variant is absent (see InstantIDAdapter).
        "variant": "fp16",
        # RealVisXL_V5.0: photoreal SDXL finetune (openrail++, commercial-OK) that
        # solves the "shiny/plastic" look; the sc-2009 A/B winner over base SDXL.
        "repo": "SG161222/RealVisXL_V5.0",
        "adapter": "instantid_sdxl",
        # InstantID IdentityNet ControlNet + ip-adapter image projection. Fetched on
        # demand by the adapter (mirrors the Kolors IP-Adapter pattern); the
        # antelopev2 face pack is fetched separately into the insightface root.
        "instantId": {
            "repo": "InstantX/InstantID",
            "controlnetSubfolder": "ControlNetModel",
            "ipAdapter": "ip-adapter.bin",
        },
        # Full-body pose set (advanced.bodyPoseSet): an OpenPose ControlNet drives the
        # standing pose while IdentityNet anchors the face. xinsir SDXL OpenPose is
        # Apache-2.0 and the strongest open one; fetched on demand (sc-2065).
        "openPose": {
            "repo": "xinsir/controlnet-openpose-sdxl-1.0",
        },
    },
    "pulid_flux_dev": {
        "label": "PuLID-FLUX (FLUX.1 [dev])",
        # FLUX-family backbone so it shares FLUX LoRA gating + the family-aware
        # preset matcher; Character Studio reference only — no plain text-to-image
        # or edit_image (the adapter requires a reference and a detectable face).
        "family": "flux",
        "supportsEdit": False,
        # sc-2012 spike defaults (PuLID "photoreal" preset): 30 steps at guidance
        # 4.0, T5 max_seq_len 128, id_weight=1.0 (the adapter knob) and
        # timestep_to_start_cfg=4. The spike measured 0.8016 ArcFace cosine vs the
        # Kelsie reference at these settings on MPS bf16 / 1024×1024 (~127 s/image,
        # ~85 GB peak unified memory). FLUX.1-dev NC license — same posture as
        # the base flux_dev built-in (already NC + gated).
        "steps": 30,
        "guidanceScale": 4.0,
        "maxSequenceLength": 128,
        "repo": "black-forest-labs/FLUX.1-dev",
        "adapter": "pulid_flux",
        # PuLID-FLUX adapter weights (the IDFormer + PerceiverAttention cross-attn
        # blocks injected into FLUX's DiT). bflConfig keys the BFL flow loader's
        # config dict in flux/util.py — only "flux-dev" is wired through today.
        "pulidFlux": {
            "repo": "guozinan/PuLID",
            "weight": "pulid_flux_v0.9.1.safetensors",
            "version": "v0.9.1",
            "bflConfig": "flux-dev",
            "maxSequenceLength": 128,
        },
    },
}


@dataclass(frozen=True)
class UpscaleRequest:
    enabled: bool = False
    factor: int = 2
    engine: str = "real-esrgan"


@dataclass(frozen=True)
class ImageRequest:
    project_id: str
    mode: str
    prompt: str
    negative_prompt: str
    model: str
    count: int
    seed: int | None
    seeds: list[int]
    width: int
    height: int
    style_preset: str
    loras: list[dict[str, Any]]
    character_id: str | None
    character_look_id: str | None
    source_asset_id: str | None
    reference_asset_id: str | None
    advanced: dict[str, Any]
    model_manifest_entry: dict[str, Any]
    upscale: UpscaleRequest = field(default_factory=UpscaleRequest)


class ImageUpscaler(Protocol):
    id: str

    def upscale(
        self,
        image: Image.Image,
        *,
        request: ImageRequest,
        cancel_requested: CancelCallback,
    ) -> Image.Image: ...


def upscale_request_from_payload(payload: dict[str, Any]) -> UpscaleRequest:
    raw = payload.get("upscale")
    if not isinstance(raw, dict):
        return UpscaleRequest()
    enabled = bool(raw.get("enabled", False))
    factor = safe_int(raw.get("factor"), 2, 2, 4)
    if factor not in {2, 4}:
        factor = 2
    engine = str(raw.get("engine") or "real-esrgan").strip() or "real-esrgan"
    return UpscaleRequest(enabled=enabled, factor=factor, engine=engine)


def image_request_from_job(job: dict[str, Any]) -> ImageRequest:
    payload = job["payload"]
    return ImageRequest(
        project_id=payload["projectId"],
        mode=payload.get("mode", "text_to_image"),
        prompt=payload.get("prompt", ""),
        negative_prompt=payload.get("negativePrompt", ""),
        model=payload.get("model", "z_image_turbo"),
        count=safe_int(payload.get("count"), 4, 1, 8),
        seed=payload.get("seed"),
        seeds=[int(seed) for seed in payload.get("seeds", []) if seed is not None],
        # Backstop only — per-model resolution is governed by manifest limits + the UI.
        # SenseNova-U1's trained buckets reach 3456 (the adapter snaps by aspect ratio),
        # so this clamp must allow the requested ratio through rather than truncate it.
        width=safe_int(payload.get("width"), 1024, 256, 4096),
        height=safe_int(payload.get("height"), 1024, 256, 4096),
        style_preset=payload.get("stylePreset", "cinematic"),
        loras=payload.get("loras", []),
        character_id=payload.get("characterId"),
        character_look_id=payload.get("characterLookId"),
        source_asset_id=payload.get("sourceAssetId"),
        reference_asset_id=payload.get("referenceAssetId"),
        model_manifest_entry=(
            payload.get("modelManifestEntry") if isinstance(payload.get("modelManifestEntry"), dict) else {}
        ),
        upscale=upscale_request_from_payload(payload),
        advanced=payload.get("advanced", {}),
    )


class ImageAssetWriter:
    def write_outputs(
        self,
        *,
        request: ImageRequest,
        project_path: Path,
        images: list[Image.Image],
        adapter_id: str,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
        raw_settings: dict[str, Any],
        settings: WorkerSettings | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return self.write_incremental_outputs(
            request=request,
            project_path=project_path,
            image_count=len(images),
            image_at_index=lambda index: images[index],
            adapter_id=adapter_id,
            progress=progress,
            cancel_requested=cancel_requested,
            raw_settings=raw_settings,
            settings=settings,
            job_id=job_id,
        )

    def write_incremental_outputs(
        self,
        *,
        request: ImageRequest,
        project_path: Path,
        image_count: int,
        image_at_index: Callable[[int], Image.Image],
        adapter_id: str,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
        raw_settings: dict[str, Any],
        settings: WorkerSettings | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        created_at = utc_now()
        generation_set_id = f"genset_{uuid4().hex}"
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        prompt_slug = slugify(request.prompt, fallback="image", max_length=42)
        date_slug = created_at[:10]
        # Each generation set saves its PNGs into its own subfolder so two jobs that
        # share the same date + model + prompt + image index cannot collide on a flat
        # `<date>_<model>_<prompt>_<index>.png` name and clobber each other's PNGs.
        # The folder carries the uniqueness (a full UUID), so the per-image filenames
        # stay short and readable. Asset discovery is rglob-based and paths are stored
        # in the sidecar/DB, so nesting is transparent downstream.
        images_dir = project_path / "assets" / "images" / generation_set_id
        images_dir.mkdir(parents=True, exist_ok=True)

        # Rust is the single project-store writer now (story 1656): the worker saves
        # only the PNG bytes and reports flat facts; the Rust API builds + writes the
        # sidecar / generation-set / recipe and indexes project.db on each progress
        # update, then re-injects the built assets into the result. We still emit a
        # progress update per image so multi-image batches keep streaming into the UI.
        generation_set = {
            "id": generation_set_id,
            "mode": request.mode,
            "model": request.model,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "count": image_count,
            "createdAt": created_at,
        }
        asset_writes: list[dict[str, Any]] = []
        upscaler = create_image_upscaler(request, settings=settings, job_id=job_id)
        expected_asset_count = image_count * (2 if upscaler is not None else 1)

        def append_image_write(
            *,
            image: Image.Image,
            asset_id: str,
            filename: str,
            seed: int,
            index: int,
            source_asset_id: str | None,
            raw_adapter_settings: dict[str, Any],
            display_suffix: str = "",
            parents: list[str] | None = None,
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            media_rel = f"assets/images/{generation_set_id}/{filename}"
            image.save(project_path / media_rel, "PNG")
            write = {
                "assetId": asset_id,
                "mediaPath": media_rel,
                "mimeType": "image/png",
                # True saved pixel dimensions, not the request's. SenseNova-U1 (and
                # any model that snaps to a trained bucket) saves at a size that
                # differs from request.width/height.
                "width": image.width,
                "height": image.height,
                "normalizedWidth": request.width,
                "normalizedHeight": request.height,
                "count": request.count,
                "family": model_target["family"],
                "seed": seed,
                "index": index,
                "displayName": f"{request.prompt[:56] or 'Generated image'} #{index + 1}{display_suffix}",
                "createdAt": created_at,
                "mode": request.mode,
                "model": request.model,
                "adapter": adapter_id,
                "prompt": request.prompt,
                "negativePrompt": request.negative_prompt,
                "loras": request.loras,
                "stylePreset": request.style_preset,
                "characterId": request.character_id,
                "characterLookId": request.character_look_id,
                "sourceAssetId": source_asset_id,
                "rawAdapterSettings": raw_adapter_settings,
            }
            if parents is not None:
                write["parents"] = parents
            if extra is not None:
                write["extra"] = extra
            asset_writes.append(write)
            return write

        for index in range(image_count):
            if cancel_requested():
                raise InterruptedError("Image generation canceled by user.")

            image = image_at_index(index)
            if cancel_requested():
                raise InterruptedError("Image generation canceled by user.")
            source_width = image.width
            source_height = image.height
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            original_asset_id = f"asset_{uuid4().hex}"
            original_filename = f"{date_slug}_{request.model}_{prompt_slug}_{index + 1:04d}.png"
            append_image_write(
                image=image,
                asset_id=original_asset_id,
                filename=original_filename,
                seed=seed,
                index=index,
                source_asset_id=request.source_asset_id,
                raw_adapter_settings=dict(raw_settings),
            )

            if upscaler is not None:
                progress(
                    "running",
                    "upscaling",
                    image_batch_progress(index, image_count),
                    f"Upscaling image {index + 1} of {image_count}.",
                )
                upscaled_image = upscaler.upscale(image, request=request, cancel_requested=cancel_requested)
                if cancel_requested():
                    raise InterruptedError("Image generation canceled by user.")
                upscale_settings: dict[str, Any] = {
                    "enabled": True,
                    "engine": upscaler.id,
                    "factor": request.upscale.factor,
                    "sourceWidth": source_width,
                    "sourceHeight": source_height,
                    "width": upscaled_image.width,
                    "height": upscaled_image.height,
                }
                upscaled_raw_settings = dict(raw_settings)
                upscaled_raw_settings["upscale"] = upscale_settings
                append_image_write(
                    image=upscaled_image,
                    asset_id=f"asset_{uuid4().hex}",
                    filename=(
                        f"{date_slug}_{request.model}_{prompt_slug}_{index + 1:04d}"
                        f"_upscaled_x{request.upscale.factor}.png"
                    ),
                    seed=seed,
                    index=index,
                    source_asset_id=original_asset_id,
                    raw_adapter_settings=upscaled_raw_settings,
                    display_suffix=f" ({request.upscale.factor}x upscaled)",
                    parents=[original_asset_id],
                    extra={
                        "isUpscaled": True,
                        "upscaledFromAssetId": original_asset_id,
                        "factor": request.upscale.factor,
                        "engine": upscaler.id,
                    },
                )
            progress(
                "saving",
                "saving",
                image_batch_progress(index + 1, image_count),
                f"Saved image asset {index + 1} of {image_count}.",
                {
                    "generationSetId": generation_set_id,
                    "expectedCount": expected_asset_count,
                    "adapter": adapter_id,
                    "model": request.model,
                    "generationSet": generation_set,
                    "assetWrites": list(asset_writes),
                },
            )

        return {
            "generationSetId": generation_set_id,
            "expectedCount": expected_asset_count,
            "adapter": adapter_id,
            "model": request.model,
            "generationSet": generation_set,
            "assetWrites": asset_writes,
        }


REAL_ESRGAN_MODEL_SPECS: dict[int, dict[str, Any]] = {
    2: {
        "name": "RealESRGAN_x2plus",
        "repo": "nateraw/real-esrgan",
        "file": "RealESRGAN_x2plus.pth",
    },
    4: {
        "name": "RealESRGAN_x4plus",
        "repo": "nateraw/real-esrgan",
        "file": "RealESRGAN_x4plus.pth",
    },
}

AURA_SR_MODEL_SPEC: dict[str, Any] = {
    "name": "AuraSR-v2",
    "repo": "fal/AuraSR-v2",
    "file": "model.safetensors",
    "scale": 4,
}


def create_image_upscaler(
    request: ImageRequest,
    *,
    settings: WorkerSettings | None = None,
    job_id: str | None = None,
) -> ImageUpscaler | None:
    if not request.upscale.enabled:
        return None
    engine = request.upscale.engine.strip().lower()
    if engine in {"real-esrgan", "realesrgan", "real_esrgan"}:
        return RealEsrganUpscaler(settings=settings, job_id=job_id)
    if engine in {"aura-sr", "aurasr", "aura_sr"}:
        return AuraSrUpscaler(settings=settings, job_id=job_id)
    raise RuntimeError(f"Unsupported image upscale engine: {request.upscale.engine}.")


class RealEsrganUpscaler:
    id = "real-esrgan"

    def __init__(self, *, settings: WorkerSettings | None = None, job_id: str | None = None) -> None:
        self._settings = settings
        self._job_id = job_id
        # Pure-PyTorch RRDBNet runner: reconstructs the network and loads the
        # real x2/x4 weights directly, so the image worker never imports
        # basicsr/realesrgan (-> cv2) or the torchvision compat shim (-> av).
        # That pairing is what triggered the macOS duplicate-AV-class warning
        # (sc-1919). The engine caches the loaded network per factor.
        self._engine = RealESRGANUpscaler()
        self._jobs: dict[int, UpscaleJob] = {}

    def upscale(
        self,
        image: Image.Image,
        *,
        request: ImageRequest,
        cancel_requested: CancelCallback,
    ) -> Image.Image:
        if cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        job = self._resolve_job(request)
        output = self._engine.upscale(image, job=job, settings=self._settings)
        if cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        return output

    def _resolve_job(self, request: ImageRequest) -> UpscaleJob:
        factor = request.upscale.factor
        cached = self._jobs.get(factor)
        if cached is not None:
            return cached
        spec = REAL_ESRGAN_MODEL_SPECS.get(factor)
        if spec is None:
            raise RuntimeError("Real-ESRGAN upscale factor must be 2 or 4.")
        model_path = self._resolve_model_path(request, spec)
        job = UpscaleJob(
            factor=factor,
            weights_path=model_path,
            engine="real_esrgan",
            tile_size=safe_int(request.advanced.get("upscaleTile"), 0, 0, 2048),
            tile_pad=safe_int(request.advanced.get("upscaleTilePad"), 10, 0, 256),
        )
        emit_worker_event(
            "image_upscaler_load_start",
            jobId=self._job_id,
            engine=self.id,
            factor=factor,
            model=spec["name"],
            modelPath=str(model_path),
        )
        device = self._engine.load(job, self._settings)
        self._jobs[factor] = job
        emit_worker_event(
            "image_upscaler_load_complete",
            jobId=self._job_id,
            engine=self.id,
            factor=factor,
            model=spec["name"],
            device=device,
        )
        return job

    def _resolve_model_path(self, request: ImageRequest, spec: dict[str, Any]) -> Path:
        explicit = (
            request.advanced.get("upscaleModelPath")
            or request.advanced.get("realEsrganModelPath")
            or os.getenv(f"SCENEWORKS_REALESRGAN_X{request.upscale.factor}_MODEL_PATH")
            or os.getenv("SCENEWORKS_REALESRGAN_MODEL_PATH")
        )
        if explicit:
            path = Path(str(explicit)).expanduser()
            if not path.is_file():
                raise RuntimeError(f"Real-ESRGAN model file does not exist: {path}")
            return path
        resource = self._manifest_resource(request, request.upscale.factor)
        repo = str(resource.get("repo") or spec["repo"])
        file_name = str(resource.get("file") or spec["file"])
        return self._hf_hub_download(repo, file_name)

    def _hf_hub_download(self, repo: str, file_name: str) -> Path:
        from huggingface_hub import hf_hub_download

        roots = huggingface_cache_roots(self._settings)
        first_error: Exception | None = None
        for cache_root in roots:
            try:
                return Path(
                    hf_hub_download(
                        repo_id=repo,
                        filename=file_name,
                        cache_dir=str(cache_root),
                        local_files_only=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - try the next configured cache root.
                first_error = exc

        cache_root = roots[0]
        try:
            return Path(hf_hub_download(repo_id=repo, filename=file_name, cache_dir=str(cache_root)))
        except Exception as exc:  # noqa: BLE001 - surface the download context.
            raise RuntimeError(
                f"Unable to resolve Real-ESRGAN weight {file_name} from Hugging Face repo {repo}."
            ) from (first_error or exc)

    @staticmethod
    def _manifest_resource(request: ImageRequest, factor: int) -> dict[str, Any]:
        resources = request.model_manifest_entry.get("resources", {})
        if not isinstance(resources, dict):
            return {}
        upscalers = resources.get("imageUpscalers") or resources.get("upscalers")
        if not isinstance(upscalers, dict):
            return {}
        real_esrgan = (
            upscalers.get("real-esrgan")
            or upscalers.get("realEsrgan")
            or upscalers.get("real_esrgan")
            or {}
        )
        if not isinstance(real_esrgan, dict):
            return {}
        resource = real_esrgan.get(f"x{factor}") or real_esrgan.get(str(factor)) or {}
        return resource if isinstance(resource, dict) else {}


class AuraSrUpscaler:
    id = "aura-sr"

    def __init__(self, *, settings: WorkerSettings | None = None, job_id: str | None = None) -> None:
        self._settings = settings
        self._job_id = job_id
        self._model: Any | None = None

    def upscale(
        self,
        image: Image.Image,
        *,
        request: ImageRequest,
        cancel_requested: CancelCallback,
    ) -> Image.Image:
        if cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        if request.upscale.factor != AURA_SR_MODEL_SPEC["scale"]:
            raise RuntimeError("AuraSR upscaling supports only 4x output.")
        model = self._load_model(request)
        max_batch_size = safe_int(request.advanced.get("auraSrMaxBatchSize"), 8, 1, 64)
        use_overlap = request.advanced.get("auraSrOverlap", request.advanced.get("upscaleOverlap", True)) is not False
        if use_overlap and hasattr(model, "upscale_4x_overlapped"):
            output = model.upscale_4x_overlapped(image.convert("RGB"), max_batch_size=max_batch_size)
        else:
            output = model.upscale_4x(image.convert("RGB"), max_batch_size=max_batch_size)
        if cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        return output.convert("RGB")

    def _load_model(self, request: ImageRequest) -> Any:
        if self._model is not None:
            return self._model
        try:
            torch = importlib.import_module("torch")
            aura_sr = importlib.import_module("aura_sr")
        except Exception as exc:  # noqa: BLE001 - convert optional dependency failures.
            raise RuntimeError(
                "AuraSR upscaling requires the optional worker package `aura-sr` to be installed."
            ) from exc

        device = select_torch_device(torch, self._settings.gpu_id if self._settings else None)
        activate_torch_device(torch, device)
        model_path = self._resolve_model_path(request)
        emit_worker_event(
            "image_upscaler_load_start",
            jobId=self._job_id,
            engine=self.id,
            factor=request.upscale.factor,
            model=AURA_SR_MODEL_SPEC["name"],
            modelPath=str(model_path),
            device=device,
        )
        self._model = aura_sr.AuraSR.from_pretrained(str(model_path), use_safetensors=True)
        upsampler = getattr(self._model, "upsampler", None)
        if upsampler is not None and hasattr(upsampler, "to"):
            upsampler.to(device)
        if upsampler is not None and hasattr(upsampler, "eval"):
            upsampler.eval()
        emit_worker_event(
            "image_upscaler_load_complete",
            jobId=self._job_id,
            engine=self.id,
            factor=request.upscale.factor,
            model=AURA_SR_MODEL_SPEC["name"],
            device=device,
        )
        return self._model

    def _resolve_model_path(self, request: ImageRequest) -> Path:
        explicit = (
            request.advanced.get("upscaleModelPath")
            or request.advanced.get("auraSrModelPath")
            or os.getenv("SCENEWORKS_AURASR_MODEL_PATH")
        )
        if explicit:
            path = Path(str(explicit)).expanduser()
            if path.is_dir():
                path = path / str(AURA_SR_MODEL_SPEC["file"])
            if not path.is_file():
                raise RuntimeError(f"AuraSR model file does not exist: {path}")
            config_path = path.with_name("config.json")
            if not config_path.is_file():
                raise RuntimeError(f"AuraSR config file does not exist next to model file: {config_path}")
            return path
        resource = self._manifest_resource(request)
        repo = str(resource.get("repo") or AURA_SR_MODEL_SPEC["repo"])
        file_name = str(resource.get("file") or AURA_SR_MODEL_SPEC["file"])
        return self._hf_snapshot_file(repo, file_name)

    def _hf_snapshot_file(self, repo: str, file_name: str) -> Path:
        from huggingface_hub import snapshot_download

        allow_patterns = [file_name, "config.json", "LICENSE.md", "README.md"]
        roots = huggingface_cache_roots(self._settings)
        first_error: Exception | None = None
        for cache_root in roots:
            try:
                snapshot_dir = Path(
                    snapshot_download(
                        repo_id=repo,
                        allow_patterns=allow_patterns,
                        cache_dir=str(cache_root),
                        local_files_only=True,
                    )
                )
                return snapshot_dir / file_name
            except Exception as exc:  # noqa: BLE001 - try the next configured cache root.
                first_error = exc

        cache_root = roots[0]
        try:
            snapshot_dir = Path(
                snapshot_download(repo_id=repo, allow_patterns=allow_patterns, cache_dir=str(cache_root))
            )
            return snapshot_dir / file_name
        except Exception as exc:  # noqa: BLE001 - surface the download context.
            raise RuntimeError(f"Unable to resolve AuraSR weight {file_name} from Hugging Face repo {repo}.") from (
                first_error or exc
            )

    @staticmethod
    def _manifest_resource(request: ImageRequest) -> dict[str, Any]:
        resources = request.model_manifest_entry.get("resources", {})
        if not isinstance(resources, dict):
            return {}
        upscalers = resources.get("imageUpscalers") or resources.get("upscalers")
        if not isinstance(upscalers, dict):
            return {}
        aura_sr = upscalers.get("aura-sr") or upscalers.get("auraSr") or upscalers.get("aura_sr") or {}
        if not isinstance(aura_sr, dict):
            return {}
        resource = aura_sr.get("x4") or aura_sr.get("4") or {}
        return resource if isinstance(resource, dict) else {}


class ZImageDiffusersAdapter:
    id = "z_image_diffusers"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._img2img_pipe: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_model: str | None = None
        self._loaded_lora_states: dict[str, LoraPipelineState] = {}

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._loaded_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        """Free any resident pipeline so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._text_pipe is None and self._img2img_pipe is None and getattr(self, "_pose_pipe", None) is None:
            return False
        self._evict_pipelines(importlib.import_module("torch"))
        return True

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
        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a Z-Image Diffusers target.")

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job["id"],
            adapter=self.id,
            loraCount=len(request.loras),
        )
        self._apply_loras(pipe, request)
        emit_worker_event("image_lora_apply_complete", jobId=job["id"], adapter=self.id)
        total = request.count
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = "Z-Image"

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
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            try:
                image = self._run_pipeline(settings, pipe, request, seed, project_path, cancel_requested=cancel_requested)
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
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
                "repo": model_target["repo"],
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request),
                "realModelInference": True,
            },
            settings=settings,
            job_id=job["id"],
        )

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
        diffusers = importlib.import_module("diffusers")
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        use_img2img = request.mode == "edit_image"
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        cached_pipe = self._img2img_pipe if use_img2img else self._text_pipe
        if cached_pipe is not None and self._loaded_repo == repo:
            self._loaded_model = request.model
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                model=request.model,
                repo=repo,
                device=device,
                componentDevices=pipeline_component_devices(cached_pipe),
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return cached_pipe

        if self._loaded_repo and self._loaded_repo != repo:
            self._evict_pipelines(torch)
        elif use_img2img and self._text_pipe is not None:
            self._text_pipe = None
            self._forget_loaded_loras("text")
            self._empty_cuda_cache(torch)
        elif not use_img2img and self._img2img_pipe is not None:
            self._img2img_pipe = None
            self._forget_loaded_loras("img2img")
            self._empty_cuda_cache(torch)

        pipeline_name = "ZImageImg2ImgPipeline" if use_img2img else "ZImagePipeline"
        pipeline_class = getattr(diffusers, pipeline_name, None)
        if pipeline_class is None and use_img2img:
            raise RuntimeError(
                "The installed diffusers package does not expose ZImageImg2ImgPipeline. "
                "Install the latest diffusers build for Z-Image edit support."
            )
        if pipeline_class is None:
            pipeline_class = getattr(diffusers, "DiffusionPipeline")

        cache_action = "Loading cached" if huggingface_repo_cache_exists(repo) else "Downloading"
        progress("loading_model", "loading_model", 0.2, f"{cache_action} {model_target['label']} model files.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useImg2img=use_img2img,
            cpuOffload=cpu_offload,
            cached=cache_action == "Loading cached",
        )
        pipe = pipeline_class.from_pretrained(
            repo,
            torch_dtype=dtype,
            low_cpu_mem_usage=bool(request.advanced.get("lowCpuMemUsage", False)),
        )
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        progress("loading_model", "loading_model", 0.22, f"Moving {model_target['label']} to {device}.")
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        component_devices = verify_pipeline_on_device(
            pipe,
            requested_device=device,
            model_label=model_target["label"],
            allow_offload=offload_enabled,
        )
        emit_worker_event(
            "image_pipeline_on_device",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            requestedDevice=device,
            cpuOffload=offload_enabled,
            componentDevices=component_devices,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

        if use_img2img:
            self._img2img_pipe = pipe
        else:
            self._text_pipe = pipe
        self._loaded_repo = repo
        self._loaded_model = request.model
        return pipe

    def _evict_pipelines(self, torch: Any) -> None:
        self._text_pipe = None
        self._img2img_pipe = None
        self._loaded_repo = None
        self._loaded_model = None
        self._loaded_lora_states.clear()
        self._empty_cuda_cache(torch)

    def _empty_cuda_cache(self, torch: Any) -> None:
        # gc.collect() first: the pipeline we just dropped is held alive by its
        # nn.Module reference cycles until the cyclic collector runs, so a bare
        # empty_cache() would reclaim nothing on MPS.
        release_inference_memory(torch)

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
        generator_device = device if device.startswith("cuda") else "cpu"
        generator = torch.Generator(generator_device).manual_seed(seed)
        sampler_key, scheduler_key, shift_value = sampler_selection_from_advanced(request.advanced)
        apply_sampler(pipe, sampler_key, scheduler_key, shift_value, adapter=self.id)
        kwargs = {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": self._num_inference_steps(request, MODEL_TARGETS[request.model]),
            "guidance_scale": self._guidance_scale(request),
            "generator": generator,
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.mode == "edit_image":
            kwargs["image"] = load_source_image(project_path, request)
            kwargs["strength"] = float(request.advanced.get("strength", 0.6))
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**kwargs)
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        image = output.images[0]
        return image.convert("RGB")

    def _apply_loras(self, pipe: Any, request: ImageRequest) -> None:
        key = "img2img" if request.mode == "edit_image" else "text"
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        self._loaded_lora_states[key] = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            previous_state=self._loaded_lora_states.get(key),
        )

    def _forget_loaded_loras(self, key: str) -> None:
        self._loaded_lora_states.pop(key, None)

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"] + 1, 1, 80)

    def _guidance_scale(self, request: ImageRequest) -> float:
        try:
            return float(request.advanced.get("guidanceScale", 0.0))
        except (TypeError, ValueError):
            return 0.0


class QwenImageAdapter:
    id = "qwen_image"

    # sc-2160: all edit IDs route through QwenImageEditPlusPipeline now that the
    # legacy qwen_image_edit / qwen_image_edit_2509 IDs alias to the 2511 base
    # (same Plus shape). Both pipelines accept `image=` + true_cfg_scale; the
    # Plus variant is the multi-image-capable one used by all modern releases.
    _EDIT_PIPELINE_BY_MODEL = {
        "qwen_image_edit": "QwenImageEditPlusPipeline",
        "qwen_image_edit_2509": "QwenImageEditPlusPipeline",
        "qwen_image_edit_2511": "QwenImageEditPlusPipeline",
        "qwen_image_edit_2511_lightning": "QwenImageEditPlusPipeline",
    }
    _DEFAULT_EDIT_PIPELINE = "QwenImageEditPipeline"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._edit_pipe: Any | None = None
        self._text_repo: str | None = None
        self._edit_repo: str | None = None
        # sc-2160: distill LoRA fuse state ?? pipeline cache must discriminate
        # between fused (Lightning) and unfused base loads on the same repo.
        self._text_distill_key: str | None = None
        self._edit_distill_key: str | None = None
        self._loaded_model: str | None = None
        self._loaded_lora_states: dict[str, LoraPipelineState] = {}

    @staticmethod
    def _use_reference(request: ImageRequest) -> bool:
        # Character Studio reference path: feed the reference as the edit-style
        # `image=` kwarg and let true_cfg_scale steer prompt vs reference. Not
        # combined with edit_image (mutually exclusive — character_image is a
        # subject-variation flow, edit_image is a localized modification flow).
        return request.mode == "character_image" and bool(request.reference_asset_id)

    @classmethod
    def _edit_pipeline_name(cls, model: str) -> str:
        return cls._EDIT_PIPELINE_BY_MODEL.get(model, cls._DEFAULT_EDIT_PIPELINE)

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._text_repo, self._edit_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        """Free any resident pipeline so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._text_pipe is None and self._edit_pipe is None:
            return False
        self._text_pipe = None
        self._edit_pipe = None
        self._text_repo = None
        self._edit_repo = None
        self._text_distill_key = None
        self._edit_distill_key = None
        self._loaded_model = None
        self._loaded_lora_states.clear()
        self._empty_cuda_cache(importlib.import_module("torch"))
        return True

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
            raise RuntimeError(f"{request.model} is not a Qwen Image target.")
        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job["id"],
            adapter=self.id,
            loraCount=len(request.loras),
        )
        self._apply_loras(pipe, request)
        emit_worker_event("image_lora_apply_complete", jobId=job["id"], adapter=self.id)
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = "Qwen Image"
        # sc-2003 multi-backbone angle set: when advanced.angleSet is set on a
        # character_image request, loop the 11 canonical angles in one job — one
        # image per pack angle, each driven by the per-angle augmented prompt.
        # Mirrors the InstantID angle-set shape (sc-2050) but uses a prompt
        # augment rather than a landmark pack, since Qwen has no landmark
        # ControlNet (the spike-validated prompt-driven path, mean ArcFace
        # cosine 0.62 on Kelsie). All angles share one seed for noise-derived
        # attribute continuity (hair / wardrobe / lighting) — same trick the
        # InstantID angle set uses.
        # Best-effort pose tier (sc-2256): advanced.poses is a list of {id, keypoints}
        # from the bundled pose library — render each as an OpenPose skeleton and feed it
        # as a second edit image alongside the reference (Qwen-Image-Edit-Plus is multi-
        # image-capable), so the pose comes from the skeleton while identity comes from
        # the reference. No pose ControlNet exists for the edit model (sc-2250); this is
        # the native multi-image approximation. Pose takes precedence over angleSet.
        raw_poses = request.advanced.get("poses")
        pose_entries = [p for p in raw_poses if isinstance(p, dict)] if isinstance(raw_poses, list) else []
        pose_set = self._use_reference(request) and len(pose_entries) > 0
        pose_keypoints = [normalize_keypoints(p.get("keypoints")) for p in pose_entries] if pose_set else []
        angle_set = self._use_reference(request) and bool(request.advanced.get("angleSet")) and not pose_set
        angles = list(CHARACTER_ANGLE_SET_ORDER) if angle_set else []
        grouped = angle_set or pose_set
        if pose_set:
            total = len(pose_entries)
        elif angle_set:
            total = len(angles)
        else:
            total = request.count
        set_seed = resolve_seed(request.seed, request.prompt, 0, request.seeds)

        def image_at_index(index: int) -> Image.Image:
            seed = set_seed if grouped else resolve_seed(request.seed, request.prompt, index, request.seeds)
            angle = angles[index] if angle_set else None
            pose_id = pose_entries[index].get("id") if pose_set else None
            pose_skeleton = None
            if pose_set:
                pose_skeleton = Image.fromarray(
                    draw_bodypose(request.width, request.height, pose_keypoints[index])
                )
                prompt_override = augment_prompt_for_pose(request.prompt)
            else:
                prompt_override = augment_prompt_for_angle(request.prompt, angle) if angle else None
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
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            try:
                image = self._run_pipeline(
                    settings,
                    pipe,
                    request,
                    seed,
                    project_path,
                    cancel_requested=cancel_requested,
                    prompt_override=prompt_override,
                    pose_skeleton=pose_skeleton,
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
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
                "repo": self._repo_for_request(request, model_target),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request),
                "realModelInference": True,
                **({"angleSet": True} if angle_set else {}),
                **({"poseLibrary": True} if pose_set else {}),
            },
            settings=settings,
            job_id=job["id"],
        )

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
        diffusers = importlib.import_module("diffusers")
        repo = self._repo_for_request(request, model_target)
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        use_edit_pipe = request.mode == "edit_image" or self._use_reference(request)
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        distill_lora = self._distill_lora_for(model_target)
        distill_key = self._distill_key_for(distill_lora)
        cached_pipe = self._edit_pipe if use_edit_pipe else self._text_pipe
        cached_repo = self._edit_repo if use_edit_pipe else self._text_repo
        cached_distill = self._edit_distill_key if use_edit_pipe else self._text_distill_key
        if cached_pipe is not None and cached_repo == repo and cached_distill == distill_key:
            self._loaded_model = request.model
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                model=request.model,
                repo=repo,
                device=device,
                componentDevices=pipeline_component_devices(cached_pipe),
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return cached_pipe
        if cached_pipe is not None:
            if use_edit_pipe:
                self._edit_pipe = None
                self._edit_repo = None
                self._edit_distill_key = None
            else:
                self._text_pipe = None
                self._text_repo = None
                self._text_distill_key = None
            self._empty_cuda_cache(torch)
            self._forget_loaded_loras("edit" if use_edit_pipe else "text")

        if use_edit_pipe:
            # Edit-style pipeline class is model-bound. As of sc-2160 every edit
            # ID (incl. legacy aliases) ships the multi-image Plus pipeline; the
            # original single-image class stays as the default fallback only.
            pipeline_name = self._edit_pipeline_name(request.model)
        else:
            pipeline_name = "QwenImagePipeline"
        pipeline_class = getattr(diffusers, pipeline_name, None)
        if pipeline_class is None:
            raise RuntimeError(f"The installed diffusers package does not expose {pipeline_name}. Install the latest diffusers build.")

        progress("loading_model", "loading_model", 0.2, f"Loading {model_target['label']} model files.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useImg2img=use_edit_pipe,
            useReference=self._use_reference(request),
            cpuOffload=cpu_offload,
            cached=huggingface_repo_cache_exists(repo),
        )
        pipe = pipeline_class.from_pretrained(repo, torch_dtype=dtype)
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        component_devices = verify_pipeline_on_device(
            pipe,
            requested_device=device,
            model_label=model_target["label"],
            allow_offload=offload_enabled,
        )
        emit_worker_event(
            "image_pipeline_on_device",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            requestedDevice=device,
            cpuOffload=offload_enabled,
            componentDevices=component_devices,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

        if distill_lora is not None:
            self._fuse_distill_lora(pipe, distill_lora, job_id=job_id)
        if use_edit_pipe:
            self._edit_pipe = pipe
            self._edit_repo = repo
            self._edit_distill_key = distill_key
        else:
            self._text_pipe = pipe
            self._text_repo = repo
            self._text_distill_key = distill_key
        self._loaded_model = request.model
        return pipe

    def _empty_cuda_cache(self, torch: Any) -> None:
        # gc.collect() first so the just-dropped pipeline is actually collected
        # before empty_cache() asks the allocator to return its blocks.
        release_inference_memory(torch)

    def _run_pipeline(
        self,
        settings: WorkerSettings,
        pipe: Any,
        request: ImageRequest,
        seed: int,
        project_path: Path,
        cancel_requested: CancelCallback | None = None,
        prompt_override: str | None = None,
        pose_skeleton: Image.Image | None = None,
    ) -> Image.Image:
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        sampler_key, scheduler_key, shift_value = sampler_selection_from_advanced(request.advanced)
        apply_sampler(pipe, sampler_key, scheduler_key, shift_value, adapter=self.id)
        # sc-2003 angleSet: each angle in the loop calls _run_pipeline with the
        # per-angle augmented prompt as `prompt_override`; falls back to the
        # request's prompt for single-image generation.
        effective_prompt = prompt_override if prompt_override else request.prompt
        kwargs = {
            "prompt": effective_prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": self._num_inference_steps(request, MODEL_TARGETS[request.model]),
            "guidance_scale": self._guidance_scale(request),
            "generator": generator,
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.mode == "edit_image":
            # QwenImageEditPipeline.__call__ takes `image=` + `true_cfg_scale` (its
            # CFG knob) but NOT a `strength` kwarg — the previous edit path passed
            # one and filter_call_kwargs silently dropped it (sc-2013 spike find).
            # Default true_cfg_scale per-model (Lightning needs 1.0; base 4.0).
            kwargs["image"] = load_source_image(project_path, request)
            kwargs["true_cfg_scale"] = self._true_cfg_scale_default(request)
        elif self._use_reference(request):
            # Character Studio reference path: same `image=` kwarg as edit_image but
            # the reference (vs source_asset_id) drives subject identity while the
            # prompt drives the new scene/pose. true_cfg_scale is the variation
            # knob: high (>4) leans prompt → more variation; low (~1) leans
            # reference → closer to source. negative_prompt is required for true CFG
            # to engage (falls back to a single space when blank).
            reference = load_reference_image(project_path, request.reference_asset_id)
            # Best-effort pose tier (sc-2256): pass [reference, skeleton] to the multi-
            # image edit pipeline so the rendered OpenPose skeleton steers the body pose
            # (the prompt cue from augment_prompt_for_pose tells the model which image is
            # the pose target). Single reference otherwise.
            kwargs["image"] = [reference, pose_skeleton] if pose_skeleton is not None else reference
            kwargs["true_cfg_scale"] = self._reference_true_cfg_scale(request)
            if "negative_prompt" not in kwargs:
                kwargs["negative_prompt"] = " "
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**filter_call_kwargs(pipe, kwargs))
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        return output.images[0].convert("RGB")

    def _apply_loras(self, pipe: Any, request: ImageRequest) -> None:
        key = "edit" if request.mode == "edit_image" else "text"
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["qwen_image"])
        self._loaded_lora_states[key] = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            previous_state=self._loaded_lora_states.get(key),
        )

    def _forget_loaded_loras(self, key: str) -> None:
        self._loaded_lora_states.pop(key, None)

    def _repo_for_request(self, request: ImageRequest, model_target: dict[str, Any]) -> str:
        return request.advanced.get("modelRepo") or model_target["repo"]

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest) -> float:
        # Default differs by target: text-to-image qwen_image uses 4.0; the Edit
        # / Edit-Plus pipelines (incl. Lightning distill) require 1.0 per the
        # model card. Per-target default lives in MODEL_TARGETS.guidanceScale.
        model_target = MODEL_TARGETS.get(getattr(request, "model", None) or "", {})
        default = model_target.get("guidanceScale", 4.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _distill_lora_for(model_target: dict[str, Any]) -> dict[str, Any] | None:
        distill = model_target.get("distillLora")
        return distill if isinstance(distill, dict) else None

    @staticmethod
    def _distill_key_for(distill_lora: dict[str, Any] | None) -> str | None:
        if distill_lora is None:
            return None
        return f"{distill_lora.get('repo', '')}/{distill_lora.get('file', '')}"

    def _fuse_distill_lora(self, pipe: Any, distill_lora: dict[str, Any], *, job_id: str) -> None:
        repo = str(distill_lora["repo"])
        file_name = str(distill_lora.get("file") or "")
        emit_worker_event(
            "image_distill_lora_fuse_start",
            jobId=job_id,
            adapter=self.id,
            repo=repo,
            file=file_name,
        )
        load_kwargs: dict[str, Any] = {}
        if file_name:
            load_kwargs["weight_name"] = file_name
        # load_lora_weights ?? fuse_lora ?? unload_lora_weights bakes the distill
        # into the base so user LoRAs can still occupy the adapter slot via the
        # existing apply_loras_to_pipeline path.
        pipe.load_lora_weights(repo, **load_kwargs)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        emit_worker_event(
            "image_distill_lora_fuse_complete",
            jobId=job_id,
            adapter=self.id,
            repo=repo,
            file=file_name,
        )

    def _true_cfg_scale_default(self, request: ImageRequest) -> float:
        # Per-model default for true_cfg_scale. Base Qwen Edit uses 4.0; Lightning
        # distill uses 1.0 (CFG disabled). MODEL_TARGETS.trueCfgScale overrides.
        model_target = MODEL_TARGETS.get(getattr(request, "model", None) or "", {})
        default = model_target.get("trueCfgScale", 4.0)
        try:
            return float(request.advanced.get("trueCfgScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _reference_true_cfg_scale(self, request: ImageRequest) -> float:
        # Variation knob for the character_image reference path. Defaults per-
        # model (Lightning fixes at 1.0, base 4.0; higher = more prompt-driven,
        # lower = closer to reference). Clamped [1, 10] — below 1 disables CFG
        # (Qwen's edit pipeline needs it >1 with a non-empty negative prompt to
        # function), above 10 collapses to pure negative-prompt steering.
        return max(1.0, min(10.0, self._true_cfg_scale_default(request)))


class FluxDiffusersAdapter:
    """Black Forest Labs FLUX.1 [schnell] / [dev] text-to-image via diffusers.FluxPipeline.

    Mirrors ZImageDiffusersAdapter / QwenImageAdapter: HF-cache check, device/dtype
    selection, progress + cancel callbacks, incremental asset writing, and worker
    events for pipeline load + inference. FLUX.1 runs in the MAIN worker venv
    (transformers 4.57 + diffusers, confirmed by spike sc-1781) — no sidecar.
    Text-to-image only; FLUX.1 Kontext editing is a future epic.
    """

    id = "flux_diffusers"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._text_repo: str | None = None
        self._loaded_model: str | None = None
        # Whether the resident pipe has the IP-Adapter (+ image encoder) loaded.
        # A plain-T2I pipe and an IP-Adapter pipe are not interchangeable, so this
        # is part of the cache key (mirrors KolorsDiffusersAdapter / SdxlDiffusersAdapter).
        self._text_ip_adapter: bool = False
        self._loaded_lora_states: dict[str, LoraPipelineState] = {}

    @staticmethod
    def _use_ip_adapter(request: ImageRequest) -> bool:
        # FLUX is T2I-only today (no edit_image); the reference branch still gates
        # on mode for parity with the SDXL/Kolors templates and to future-proof
        # FLUX.1 Kontext when it lands.
        return request.mode != "edit_image" and bool(request.reference_asset_id)

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._text_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        """Free the resident pipeline so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._text_pipe is None:
            return False
        self._text_pipe = None
        self._text_repo = None
        self._text_ip_adapter = False
        self._loaded_model = None
        self._loaded_lora_states.clear()
        self._empty_cuda_cache(importlib.import_module("torch"))
        return True

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
            raise RuntimeError(f"{request.model} is not a FLUX.1 Diffusers target.")
        if request.mode == "edit_image":
            raise RuntimeError(f"{request.model} does not support image editing.")

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job["id"],
            adapter=self.id,
            loraCount=len(request.loras),
        )
        self._apply_loras(pipe, request)
        emit_worker_event("image_lora_apply_complete", jobId=job["id"], adapter=self.id)
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
                gpuMemory=gpu_memory_snapshot(torch, device),
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
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
                "repo": self._repo_for_request(request, model_target),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request, model_target),
                "maxSequenceLength": self._max_sequence_length(request, model_target),
                "realModelInference": True,
            },
        )

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
        diffusers = importlib.import_module("diffusers")
        repo = self._repo_for_request(request, model_target)
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        use_ip_adapter = self._use_ip_adapter(request)
        ip_adapter = model_target.get("ipAdapter") or {}
        if use_ip_adapter and not ip_adapter:
            raise RuntimeError(
                f"{request.model} does not support reference-image (IP-Adapter) generation."
            )
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        if (
            self._text_pipe is not None
            and self._text_repo == repo
            and self._text_ip_adapter == use_ip_adapter
        ):
            self._loaded_model = request.model
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                model=request.model,
                repo=repo,
                device=device,
                componentDevices=pipeline_component_devices(self._text_pipe),
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return self._text_pipe
        if self._text_pipe is not None:
            self._text_pipe = None
            self._text_repo = None
            self._text_ip_adapter = False
            self._empty_cuda_cache(torch)
            self._forget_loaded_loras("text")

        pipeline_class = getattr(diffusers, "FluxPipeline", None)
        if pipeline_class is None:
            raise RuntimeError(
                "The installed diffusers package does not expose FluxPipeline. "
                "Install diffusers >= 0.30 for FLUX.1 support."
            )

        progress("loading_model", "loading_model", 0.2, f"Loading {model_target['label']} model files.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useImg2img=False,
            useIpAdapter=use_ip_adapter,
            cpuOffload=cpu_offload,
            cached=huggingface_repo_cache_exists(repo),
        )
        pipe = pipeline_class.from_pretrained(repo, torch_dtype=dtype)
        if use_ip_adapter:
            # FluxIPAdapterMixin.load_ip_adapter takes the image-encoder repo/subfolder
            # directly (no separate CLIPVisionModelWithProjection.from_pretrained step),
            # unlike SDXL/Kolors. XLabs default = openai/clip-vit-large-patch14.
            pipe.load_ip_adapter(
                ip_adapter["repo"],
                weight_name=ip_adapter["weight"],
                subfolder=ip_adapter.get("subfolder", ""),
                image_encoder_pretrained_model_name_or_path=ip_adapter["imageEncoderRepo"],
                image_encoder_subfolder=ip_adapter.get("imageEncoderSubfolder", ""),
                image_encoder_dtype=dtype,
            )
            pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        # VAE tiling keeps high-resolution decodes within memory. Prefer the
        # current diffusers API (pipe.vae.enable_tiling) and fall back to the
        # deprecated pipeline-level shim for older builds.
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        elif hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        component_devices = verify_pipeline_on_device(
            pipe,
            requested_device=device,
            model_label=model_target["label"],
            allow_offload=offload_enabled,
        )
        emit_worker_event(
            "image_pipeline_on_device",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            requestedDevice=device,
            cpuOffload=offload_enabled,
            componentDevices=component_devices,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        self._text_pipe = pipe
        self._text_repo = repo
        self._text_ip_adapter = use_ip_adapter
        self._loaded_model = request.model
        return pipe

    def _empty_cuda_cache(self, torch: Any) -> None:
        release_inference_memory(torch)

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
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        model_target = MODEL_TARGETS[request.model]
        kwargs = {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": self._num_inference_steps(request, model_target),
            # FLUX uses embedded (distilled) guidance: schnell ignores it (0.0),
            # dev follows it (~3.5). Real CFG against negative_prompt rides on the
            # parallel true_cfg_scale kwarg, set by the IP-Adapter branch below.
            "guidance_scale": self._guidance_scale(request, model_target),
            "max_sequence_length": self._max_sequence_length(request, model_target),
            "generator": generator,
        }
        if self._use_ip_adapter(request):
            # IP-Adapter conditions T2I on a reference image. FLUX is guidance-
            # distilled, so the diffusers FLUX pipeline exposes true_cfg_scale to
            # turn real classifier-free guidance against negative_prompt back on
            # for the duration of the IP-Adapter run (XLabs default ~4.0).
            kwargs["ip_adapter_image"] = load_reference_image(project_path, request.reference_asset_id)
            kwargs["negative_prompt"] = request.negative_prompt
            kwargs["true_cfg_scale"] = self._true_cfg_scale(request)
            if hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**filter_call_kwargs(pipe, kwargs))
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        return output.images[0].convert("RGB")

    def _apply_loras(self, pipe: Any, request: ImageRequest) -> None:
        model_target = MODEL_TARGETS.get(request.model, {})
        self._loaded_lora_states["text"] = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            previous_state=self._loaded_lora_states.get("text"),
        )

    def _forget_loaded_loras(self, key: str) -> None:
        self._loaded_lora_states.pop(key, None)

    def _repo_for_request(self, request: ImageRequest, model_target: dict[str, Any]) -> str:
        return request.advanced.get("modelRepo") or model_target["repo"]

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = model_target.get("guidanceScale", 0.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _max_sequence_length(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(
            request.advanced.get("maxSequenceLength"),
            model_target.get("maxSequenceLength", 512),
            1,
            512,
        )

    def _ip_adapter_scale(self, request: ImageRequest) -> float:
        # How strongly the reference conditions the result (0 = ignore, 1 = maximal).
        # XLabs+CLIP-L is a resemblance tier; 0.7 holds composition/style while
        # leaving the prompt headroom. Faithful face identity belongs to PuLID-FLUX
        # (sc-2012), not this engine.
        try:
            scale = float(request.advanced.get("ipAdapterScale", 0.7))
        except (TypeError, ValueError):
            return 0.7
        return max(0.0, min(1.0, scale))

    def _true_cfg_scale(self, request: ImageRequest) -> float:
        # FLUX is guidance-distilled, so its `guidance_scale` is the distilled
        # signal baked into the model — it does NOT do CFG against the negative
        # prompt. With IP-Adapter, the diffusers FluxPipeline exposes the parallel
        # `true_cfg_scale` kwarg that re-enables real classifier-free guidance for
        # the duration of the run. XLabs docs default ~4.0 (range ~1.0 – 6.0).
        try:
            scale = float(request.advanced.get("trueCfgScale", 4.0))
        except (TypeError, ValueError):
            return 4.0
        return max(1.0, min(10.0, scale))


class MlxFluxAdapter:
    """FLUX.1 [schnell] / [dev] text-to-image via mflux (Apple MLX), run OUT-OF-PROCESS.

    mflux 0.17.5 hard-requires transformers>=5 + huggingface_hub>=1, which conflict
    with the main worker venv's transformers 4.57.x + huggingface_hub<1 (held there
    for native LTX-2.3 and the existing FluxDiffusersAdapter / Wan / Kolors / etc.
    diffusers paths). So mflux runs in a dedicated sidecar venv
    (`/opt/mlx-flux-venv`) via ``scene_worker/mlx_flux_runner.py``; this adapter
    only orchestrates that subprocess and writes the resulting PNGs through the
    shared asset writer. Mirrors LensTurboAdapter (same dep-divergence pattern).

    Spike sc-1969 (2026-05-28, M5 Max 128 GB) verdict: GO. mflux bf16 ~2.84 s/step
    vs torch/MPS FluxDiffusersAdapter ~4.06 s/step (~30% faster) at 1024² 4-step.
    Q4 and Q8 match bf16 speed on M-series (mlx ops dominate over memory
    bandwidth), with significant peak-memory reduction (Q4 41.6 GB vs bf16 65.8 GB).
    XLabs FLUX LoRA (152 weight keys) loaded cleanly via mflux's lora_paths plumbing.

    Selected when ALL of:
      - the request model is in ``_supported_models`` (flux_schnell, flux_dev)
      - MPS is available (``_mps_available()``)
      - the sidecar venv exists (``_sidecar_available()``)
      - ``SCENEWORKS_DISABLE_MLX_FLUX`` is unset
      - the request has no reference asset (mflux has no FLUX IP-Adapter today)

    Falls back to FluxDiffusersAdapter (torch/MPS) on any of these failing.
    Never regresses the torch path — adapter is purely additive.

    Text-to-image only for v1; mflux also covers Kontext (edit), Fill, Depth,
    Redux, ControlNet, Qwen-Image, Z-Image, FLUX.2 — out of sc-1970 scope.
    """

    id = "mlx_flux"
    _supported_models = {"flux_schnell", "flux_dev"}

    def __init__(self) -> None:
        # Sidecar scratch dir for the in-flight job, reaped by discard_temp_outputs
        # on force-cancel (os._exit skips the finally in generate). Mirrors
        # LensTurboAdapter's per-job scratch lifecycle.
        self._scratch_dir: Path | None = None

    def discard_temp_outputs(self, job_id: str | None = None) -> None:
        """Reap the in-flight sidecar scratch dir only — filesystem-only.

        Called from generate's finally and from the force-cancel monitor thread
        right before os._exit, so it must stay filesystem-only (no torch/GPU)."""
        work_dir = self._scratch_dir
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._scratch_dir = None

    def loaded_models(self) -> list[str]:
        # The sidecar process loads and frees the model per job; nothing stays
        # resident in this (main-venv) process.
        return []

    @staticmethod
    def _sidecar_python() -> str:
        return os.getenv("SCENEWORKS_MLX_FLUX_PYTHON", "/opt/mlx-flux-venv/bin/python")

    @staticmethod
    def _runner_path() -> Path:
        return Path(__file__).resolve().parent / "mlx_flux_runner.py"

    def _sidecar_available(self) -> bool:
        return Path(self._sidecar_python()).exists() and self._runner_path().exists()

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
        if model_target.get("adapter") != FluxDiffusersAdapter.id:
            raise RuntimeError(f"{request.model} is not a FLUX.1 target.")
        if request.model not in self._supported_models:
            raise RuntimeError(
                f"MlxFluxAdapter supports "
                f"{', '.join(sorted(self._supported_models))}, not {request.model}."
            )
        if request.mode == "edit_image":
            # mflux supports Kontext edit but the FLUX SceneWorks contract is
            # T2I-only today; routing edit_image here would silently change
            # behavior. The dispatch shouldn't pick MLX for edit jobs.
            raise RuntimeError(f"{request.model} MLX adapter is text-to-image only (v1).")
        if request.reference_asset_id:
            # FLUX IP-Adapter (XLabs) lives on the torch path. mflux has no
            # equivalent today, so a reference-image request must fall back.
            raise RuntimeError(
                f"{request.model} reference-image (IP-Adapter) generation is not supported on "
                "the MLX backend. Use the torch path (SCENEWORKS_IMAGE_ADAPTER=flux_diffusers)."
            )

        # Resolve + validate LoRAs in the main venv so a bad path or incompatible
        # family fails before we spawn the subprocess. The sidecar only sees
        # concrete file paths + weights.
        validate_lora_compatibility(
            request.loras, model_family=model_target.get("family"), adapter_id=self.id, model_id=request.model
        )
        lora_specs = normalize_lora_specs(request.loras)
        # The MLX backend can't apply LoKr (its merge math is LoRA-only); reject
        # clearly rather than silently ignoring the adapter (epic 2193).
        reject_lokr_loras(lora_specs, self.id)

        if not self._sidecar_available():
            raise RuntimeError(
                "MLX FLUX generation requires the isolated mlx-flux sidecar venv. The desktop "
                "bootstrap installs it on first launch (apps/desktop/src/setup.rs "
                "provision_mlx_flux_venv); rebuild without SCENEWORKS_DISABLE_MLX_FLUX=1, or set "
                "SCENEWORKS_MLX_FLUX_PYTHON to a Python interpreter that has mflux installed "
                f"(looked for {self._sidecar_python()})."
            )

        total = request.count
        steps = self._num_inference_steps(request, model_target)
        guidance = self._guidance_scale(request, model_target)
        quantize = self._resolve_quantize(request)
        seeds = [resolve_seed(request.seed, request.prompt, index, request.seeds) for index in range(total)]

        progress(
            "loading_model",
            "loading_model",
            0.18,
            f"Loading {model_target['label']} (MLX sidecar venv).",
        )
        work_dir = Path(tempfile.mkdtemp(prefix="mlx_flux_sidecar_"))
        self._scratch_dir = work_dir
        try:
            images = self._run_sidecar(
                job_id=job["id"],
                work_dir=work_dir,
                label=model_target["label"],
                total=total,
                spec={
                    "model": request.model,
                    "prompt": request.prompt,
                    "negativePrompt": request.negative_prompt or None,
                    "seeds": seeds,
                    "height": request.height,
                    "width": request.width,
                    "numInferenceSteps": steps,
                    "guidance": guidance,
                    "quantize": quantize,
                    "loras": [
                        {"path": lora.path, "weight": lora.weight, "name": lora.adapter_name}
                        for lora in lora_specs
                    ],
                },
                progress=progress,
                cancel_requested=cancel_requested,
            )

            def image_at_index(index: int) -> Image.Image:
                progress(
                    "running",
                    "generating",
                    image_batch_progress(index, total),
                    format_batch_running_message(model_target["label"], index, total),
                )
                with Image.open(images[index]) as handle:
                    return handle.convert("RGB")

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
                    "repo": model_target["repo"],
                    "numInferenceSteps": steps,
                    "guidanceScale": guidance,
                    "mlxQuantize": quantize,
                    "sidecarVenv": self._sidecar_python(),
                    "realModelInference": True,
                },
                settings=settings,
                job_id=job["id"],
            )
        finally:
            # The writer has read every PNG into the project by now; drop the
            # sidecar's scratch dir regardless of success/failure (also clears
            # the force-cancel registry).
            self.discard_temp_outputs(job["id"])

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = model_target.get("guidanceScale", 0.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _resolve_quantize(self, request: ImageRequest) -> int | None:
        """Pick the mflux quantize level. Order: advanced override > manifest
        mlx.quantize > Q8 default.

        Q8 was the sc-1969 spike sweet spot: same per-step time as bf16 on M-series,
        ~25% peak-memory reduction, no visible quality drift. Q4 is the right pick
        for ≤64 GB Macs (peak ~41.6 GB at 1024²) with minor detail drift. None
        keeps the model in bf16 (highest memory, zero quality loss).
        """
        override = request.advanced.get("mlxQuantize")
        if override is not None:
            if isinstance(override, bool):
                # Treat True/False as "no override" rather than 0/1 quant.
                pass
            else:
                try:
                    parsed = int(override)
                    return parsed if parsed > 0 else None
                except (TypeError, ValueError):
                    pass
        mlx_entry = request.model_manifest_entry.get("mlx") if request.model_manifest_entry else None
        if isinstance(mlx_entry, dict):
            manifest_q = mlx_entry.get("quantize")
            if manifest_q is not None:
                try:
                    parsed = int(manifest_q)
                    return parsed if parsed > 0 else None
                except (TypeError, ValueError):
                    pass
        return 8

    def _run_sidecar(
        self,
        *,
        job_id: str,
        work_dir: Path,
        label: str,
        total: int,
        spec: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> list[str]:
        spec = {**spec, "outDir": str(work_dir)}
        spec_path = work_dir / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        stdout_log = work_dir / "stdout.log"
        cmd = [self._sidecar_python(), str(self._runner_path()), str(spec_path)]
        emit_worker_event(
            "mlx_flux_sidecar_start",
            jobId=job_id,
            adapter=self.id,
            model=spec["model"],
            imageCount=total,
            quantize=spec.get("quantize"),
            sidecar=self._sidecar_python(),
        )
        progress(
            "running",
            "generating",
            image_batch_progress(0, total),
            f"Running {label} ({total} image(s)).",
        )
        # stdout -> file (avoids any pipe-fill deadlock); stderr inherits to the
        # worker log for diagnostics. Poll so the job stays cancelable; the
        # heartbeat thread keeps it alive during the (minutes-long) run. Mirrors
        # LensTurboAdapter._run_sidecar.
        with stdout_log.open("w", encoding="utf-8") as out:
            # stderr merged into stdout.log so a native crash (SIGABRT/SIGSEGV)
            # leaves a partial traceback we can surface; result.json stays the
            # authoritative success channel (_read_result prefers it).
            proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT)
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_requested():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise InterruptedError("Image generation canceled by user.")
        result = self._read_result(work_dir, stdout_log)
        if proc.returncode != 0 or "error" in result:
            error = result.get("error") or f"MLX FLUX sidecar exited with code {proc.returncode}."
            error = _mlx_sidecar_failure_detail(error, proc.returncode, stdout_log)
            emit_worker_event(
                "mlx_flux_sidecar_failed",
                jobId=job_id,
                adapter=self.id,
                error=error,
                returnCode=proc.returncode,
            )
            raise RuntimeError(f"MLX FLUX generation failed in the sidecar venv: {error}")
        images = [str(path) for path in result.get("images", [])]
        if len(images) != total:
            raise RuntimeError(f"MLX FLUX sidecar produced {len(images)} image(s); expected {total}.")
        emit_worker_event("mlx_flux_sidecar_complete", jobId=job_id, adapter=self.id, imageCount=len(images))
        return images

    @staticmethod
    def _read_result(work_dir: Path, stdout_log: Path) -> dict[str, Any]:
        result_path = work_dir / "result.json"
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
        return {"error": "MLX FLUX sidecar produced no parseable result."}


def _mlx_sidecar_failure_detail(error: str, returncode: int, stdout_log: Path) -> str:
    """Enrich an opaque MLX-sidecar failure with the OS exit signal + output tail.

    A *negative* return code means the OS killed the sidecar with a signal, which
    bypasses ``mlx_flux_runner``'s ``try/except`` (so it never gets to write a
    structured error — which is exactly why the only message is "produced no
    parseable result"). The signal tells us which failure mode it was:

      - ``-9`` (SIGKILL): macOS memory pressure / jetsam — i.e. out of memory.
      - ``-6`` (SIGABRT) / ``-11`` (SIGSEGV): a native MLX/Metal abort or fault.

    For the abort/fault cases a partial traceback usually lands in the captured
    output (stderr is now merged into ``stdout.log``), so we append a short tail.
    """
    parts: list[str] = [error] if error else []
    if returncode < 0:
        sig = -returncode
        label = {6: "SIGABRT", 9: "SIGKILL", 11: "SIGSEGV"}.get(sig, f"signal {sig}")
        if sig == 9:
            parts.append(
                f"sidecar killed by {label} — almost certainly out of memory; "
                "try a lower resolution, Q4 quantization, or fewer LoRAs"
            )
        else:
            parts.append(f"sidecar crashed with {label} (native MLX/Metal fault)")
    try:
        tail = "\n".join(stdout_log.read_text(encoding="utf-8").splitlines()[-15:]).strip()
    except OSError:
        tail = ""
    if tail:
        parts.append(f"last sidecar output:\n{tail}")
    return " — ".join(parts) if parts else "MLX sidecar failed with no output."


class MlxQwenAdapter:
    """Qwen-Image text-to-image via mflux (Apple MLX), run OUT-OF-PROCESS.

    Mirrors `MlxFluxAdapter` (sc-1970) — same sidecar venv (`/opt/mlx-flux-venv`),
    same `mlx_flux_runner.py` (which dispatches on `spec["model"]` to the matching
    mflux family class — `Flux1` for FLUX, `QwenImage` for Qwen). The sidecar is
    required because mflux's transformers>=5 + huggingface_hub>=1 stack cannot
    coexist with the main worker venv's transformers 4.57.x + huggingface_hub<1
    (same divergence that forced the Lens sidecar and the FLUX MLX sidecar).
    sc-1972.

    Spike measurement (sc-1969 spike venv, M5 Max 128 GB, mflux 0.17.5,
    Qwen-Image Q8, 1024² 20 steps): 7.30 s/step (~146 s wall-clock), ~65.5 GB
    peak footprint. Direct comparison vs the torch `QwenImageAdapter` baseline
    is deferred to the in-tree QA pass — the per-step cadence here is the same
    order of magnitude as the FLUX path's spike, and the win mflux provides
    is the optional Q4 lever for tighter Mac memory budgets.

    v1 scope: `qwen_image` text-to-image only. `qwen_image_edit` and
    `qwen_image_edit_2509` need spec/runner extension for `image_paths` (mflux
    `QwenImageEdit.generate_image` signature differs slightly) plus reference
    asset threading from the main venv; tracked as a follow-up.

    Selected when ALL of:
      - the request model is in ``_supported_models`` (qwen_image)
      - ``sys.platform == "darwin"``
      - the sidecar venv exists (``_sidecar_available()``)
      - ``SCENEWORKS_DISABLE_MLX_FLUX`` is unset (one env shared by the whole
        mflux sidecar — opt-out is global to the venv, not per-family)
      - the request has no reference asset and no edit_image mode (T2I-only)

    Falls back to `QwenImageAdapter` (torch / diffusers) on any of these
    failing. Never regresses the torch path.
    """

    id = "mlx_qwen"
    _supported_models = {"qwen_image"}

    def __init__(self) -> None:
        # Per-job scratch dir, reaped by discard_temp_outputs on force-cancel.
        # Mirrors MlxFluxAdapter's lifecycle exactly.
        self._scratch_dir: Path | None = None

    def discard_temp_outputs(self, job_id: str | None = None) -> None:
        work_dir = self._scratch_dir
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._scratch_dir = None

    def loaded_models(self) -> list[str]:
        return []

    @staticmethod
    def _sidecar_python() -> str:
        # Shared sidecar venv with MlxFluxAdapter (both Pythons live in
        # /opt/mlx-flux-venv). The env var is set by the desktop bootstrap
        # for every host; existence is gated by `_sidecar_available()`.
        return os.getenv("SCENEWORKS_MLX_FLUX_PYTHON", "/opt/mlx-flux-venv/bin/python")

    @staticmethod
    def _runner_path() -> Path:
        # Same runner as MlxFluxAdapter; the runner dispatches on the spec's
        # `model` field to the matching mflux family class.
        return Path(__file__).resolve().parent / "mlx_flux_runner.py"

    def _sidecar_available(self) -> bool:
        return Path(self._sidecar_python()).exists() and self._runner_path().exists()

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
        if model_target.get("adapter") != QwenImageAdapter.id:
            raise RuntimeError(f"{request.model} is not a Qwen-Image target.")
        if request.model not in self._supported_models:
            raise RuntimeError(
                f"MlxQwenAdapter supports "
                f"{', '.join(sorted(self._supported_models))}, not {request.model}."
            )
        if request.mode == "edit_image":
            raise RuntimeError(
                f"{request.model} MLX adapter is text-to-image only (v1). "
                "Set SCENEWORKS_IMAGE_ADAPTER=qwen_image for the torch edit path."
            )
        if request.reference_asset_id:
            raise RuntimeError(
                f"{request.model} reference-image (character) generation is not yet wired "
                "on the MLX backend (needs mflux QwenImageEdit.image_paths threading). "
                "Use the torch path (SCENEWORKS_IMAGE_ADAPTER=qwen_image)."
            )

        # Resolve + validate LoRAs in the main venv so a bad path or incompatible
        # family fails before we spawn the subprocess. The sidecar only sees
        # concrete file paths + weights.
        validate_lora_compatibility(
            request.loras, model_family=model_target.get("family"), adapter_id=self.id, model_id=request.model
        )
        lora_specs = normalize_lora_specs(request.loras)
        # The MLX backend can't apply LoKr (its merge math is LoRA-only); reject
        # clearly rather than silently ignoring the adapter (epic 2193).
        reject_lokr_loras(lora_specs, self.id)

        if not self._sidecar_available():
            raise RuntimeError(
                "MLX Qwen generation requires the isolated mlx-flux sidecar venv "
                "(shared with MlxFluxAdapter). The desktop bootstrap installs it on "
                "first launch (apps/desktop/src/setup.rs provision_mlx_flux_venv); "
                "rebuild without SCENEWORKS_DISABLE_MLX_FLUX=1, or set "
                "SCENEWORKS_MLX_FLUX_PYTHON to a Python interpreter that has mflux "
                f"installed (looked for {self._sidecar_python()})."
            )

        total = request.count
        steps = self._num_inference_steps(request, model_target)
        guidance = self._guidance_scale(request)
        quantize = self._resolve_quantize(request)
        seeds = [resolve_seed(request.seed, request.prompt, index, request.seeds) for index in range(total)]

        progress(
            "loading_model",
            "loading_model",
            0.18,
            f"Loading {model_target['label']} (MLX sidecar venv).",
        )
        work_dir = Path(tempfile.mkdtemp(prefix="mlx_qwen_sidecar_"))
        self._scratch_dir = work_dir
        try:
            images = self._run_sidecar(
                job_id=job["id"],
                work_dir=work_dir,
                label=model_target["label"],
                total=total,
                spec={
                    "model": request.model,
                    "prompt": request.prompt,
                    "negativePrompt": request.negative_prompt or None,
                    "seeds": seeds,
                    "height": request.height,
                    "width": request.width,
                    "numInferenceSteps": steps,
                    "guidance": guidance,
                    "quantize": quantize,
                    "loras": [
                        {"path": lora.path, "weight": lora.weight, "name": lora.adapter_name}
                        for lora in lora_specs
                    ],
                },
                progress=progress,
                cancel_requested=cancel_requested,
            )

            def image_at_index(index: int) -> Image.Image:
                progress(
                    "running",
                    "generating",
                    image_batch_progress(index, total),
                    format_batch_running_message(model_target["label"], index, total),
                )
                with Image.open(images[index]) as handle:
                    return handle.convert("RGB")

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
                    "repo": model_target["repo"],
                    "numInferenceSteps": steps,
                    "guidanceScale": guidance,
                    "mlxQuantize": quantize,
                    "sidecarVenv": self._sidecar_python(),
                    "realModelInference": True,
                },
                settings=settings,
                job_id=job["id"],
            )
        finally:
            self.discard_temp_outputs(job["id"])

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest) -> float:
        # Qwen-Image defaults to 4.0 (matches QwenImageAdapter — model-card
        # default for both the torch path and the mflux path).
        try:
            return float(request.advanced.get("guidanceScale", 4.0))
        except (TypeError, ValueError):
            return 4.0

    def _resolve_quantize(self, request: ImageRequest) -> int | None:
        """Pick the mflux quantize level. Order: advanced > manifest > Q8 default.
        Identical to MlxFluxAdapter._resolve_quantize."""
        override = request.advanced.get("mlxQuantize")
        if override is not None and not isinstance(override, bool):
            try:
                parsed = int(override)
                return parsed if parsed > 0 else None
            except (TypeError, ValueError):
                pass
        mlx_entry = request.model_manifest_entry.get("mlx") if request.model_manifest_entry else None
        if isinstance(mlx_entry, dict):
            manifest_q = mlx_entry.get("quantize")
            if manifest_q is not None:
                try:
                    parsed = int(manifest_q)
                    return parsed if parsed > 0 else None
                except (TypeError, ValueError):
                    pass
        return 8

    def _run_sidecar(
        self,
        *,
        job_id: str,
        work_dir: Path,
        label: str,
        total: int,
        spec: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> list[str]:
        spec = {**spec, "outDir": str(work_dir)}
        spec_path = work_dir / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        stdout_log = work_dir / "stdout.log"
        cmd = [self._sidecar_python(), str(self._runner_path()), str(spec_path)]
        emit_worker_event(
            "mlx_qwen_sidecar_start",
            jobId=job_id,
            adapter=self.id,
            model=spec["model"],
            imageCount=total,
            quantize=spec.get("quantize"),
            sidecar=self._sidecar_python(),
        )
        progress(
            "running",
            "generating",
            image_batch_progress(0, total),
            f"Running {label} ({total} image(s)).",
        )
        with stdout_log.open("w", encoding="utf-8") as out:
            # stderr merged into stdout.log so a native crash (SIGABRT/SIGSEGV)
            # leaves a partial traceback we can surface; result.json stays the
            # authoritative success channel (_read_result prefers it).
            proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT)
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_requested():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise InterruptedError("Image generation canceled by user.")
        result = self._read_result(work_dir, stdout_log)
        if proc.returncode != 0 or "error" in result:
            error = result.get("error") or f"MLX Qwen sidecar exited with code {proc.returncode}."
            error = _mlx_sidecar_failure_detail(error, proc.returncode, stdout_log)
            emit_worker_event(
                "mlx_qwen_sidecar_failed",
                jobId=job_id,
                adapter=self.id,
                error=error,
                returnCode=proc.returncode,
            )
            raise RuntimeError(f"MLX Qwen generation failed in the sidecar venv: {error}")
        images = [str(path) for path in result.get("images", [])]
        if len(images) != total:
            raise RuntimeError(f"MLX Qwen sidecar produced {len(images)} image(s); expected {total}.")
        emit_worker_event("mlx_qwen_sidecar_complete", jobId=job_id, adapter=self.id, imageCount=len(images))
        return images

    @staticmethod
    def _read_result(work_dir: Path, stdout_log: Path) -> dict[str, Any]:
        result_path = work_dir / "result.json"
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
        return {"error": "MLX Qwen sidecar produced no parseable result."}


class MlxZImageAdapter:
    """Z-Image-Turbo text-to-image via mflux (Apple MLX), run OUT-OF-PROCESS.

    Third mflux family on the sc-1970 sidecar venv (after sc-1972 wired Qwen).
    Same `/opt/mlx-flux-venv`, same `mlx_flux_runner.py` (dispatches on
    `spec["model"]` to the matching mflux family class — `ZImage` for
    `z_image_turbo`). Selection mirrors `MlxQwenAdapter` exactly; only the
    supported-models set, the worker-event prefix, and the runtime adapter id
    differ. sc-2145.

    Spike measurement (2026-05-28, M5 Max 128 GB, mflux 0.17.5, 1024² 8
    steps Q8): 53.3 s wall-clock, ~4.6 s/step steady-state, ~52.5 GB peak.
    Z-Image-Turbo's value is its 8-step distillation, not raw step speed.

    v1 scope: `z_image_turbo` only. `mflux.models.common.config.model_config`
    also exposes `ModelConfig.z_image()` (full / non-turbo), but SceneWorks
    only catalogs Turbo today — a follow-up can add Z-Image base if and when
    a `z_image` manifest entry lands.

    Selected when ALL of:
      - the request model is in ``_supported_models`` (z_image_turbo)
      - ``sys.platform == "darwin"``
      - the sidecar venv exists (``_sidecar_available()``)
      - ``SCENEWORKS_DISABLE_MLX_FLUX`` is unset (shared opt-out across the
        whole mflux sidecar — one switch per venv, not per family)
      - the request has no reference asset and no edit_image mode

    Falls back to ``ZImageDiffusersAdapter`` (torch / diffusers) on any of
    these failing. Never regresses the torch path.
    """

    id = "mlx_z_image"
    _supported_models = {"z_image_turbo"}

    def __init__(self) -> None:
        self._scratch_dir: Path | None = None

    def discard_temp_outputs(self, job_id: str | None = None) -> None:
        work_dir = self._scratch_dir
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._scratch_dir = None

    def loaded_models(self) -> list[str]:
        return []

    @staticmethod
    def _sidecar_python() -> str:
        # Shared sidecar venv with MlxFluxAdapter / MlxQwenAdapter — one
        # interpreter at /opt/mlx-flux-venv hosts every mflux family.
        return os.getenv("SCENEWORKS_MLX_FLUX_PYTHON", "/opt/mlx-flux-venv/bin/python")

    @staticmethod
    def _runner_path() -> Path:
        # Shared runner; dispatches on the spec's `model` field.
        return Path(__file__).resolve().parent / "mlx_flux_runner.py"

    def _sidecar_available(self) -> bool:
        return Path(self._sidecar_python()).exists() and self._runner_path().exists()

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
        if model_target.get("adapter") != ZImageDiffusersAdapter.id:
            raise RuntimeError(f"{request.model} is not a Z-Image target.")
        if request.model not in self._supported_models:
            raise RuntimeError(
                f"MlxZImageAdapter supports "
                f"{', '.join(sorted(self._supported_models))}, not {request.model}."
            )
        if request.mode == "edit_image":
            raise RuntimeError(
                f"{request.model} MLX adapter is text-to-image only. "
                "Use SCENEWORKS_IMAGE_ADAPTER=z_image_diffusers for the torch path."
            )
        # Strict pose tier (sc-2257): advanced.poses is a list of {id, keypoints}.
        # Each pose renders to an OpenPose skeleton that conditions the ported
        # Z-Image Fun-Controlnet-Union branch (true pose lock, not best-effort).
        # No reference is used — Z-Image has no IP-Adapter, so identity comes from
        # the prompt; the skeleton supplies the pose.
        raw_poses = request.advanced.get("poses")
        pose_entries = [p for p in raw_poses if isinstance(p, dict)] if isinstance(raw_poses, list) else []
        pose_set = len(pose_entries) > 0
        if request.reference_asset_id and not pose_set:
            # ZImageDiffusersAdapter itself has no reference path (sc-2005
            # cleanup); raising here keeps the error surface honest if a
            # caller manually targets MlxZImageAdapter with one. Pose requests
            # may carry a character reference we can't consume — ignore it
            # rather than reject (the skeleton drives the pose).
            raise RuntimeError(
                f"{request.model} reference-image generation is not supported "
                "on the MLX backend (Z-Image has no IP-Adapter weights upstream)."
            )

        validate_lora_compatibility(
            request.loras, model_family=model_target.get("family"), adapter_id=self.id, model_id=request.model
        )
        lora_specs = normalize_lora_specs(request.loras)
        # The MLX backend can't apply LoKr (its merge math is LoRA-only); reject
        # clearly rather than silently ignoring the adapter (epic 2193).
        reject_lokr_loras(lora_specs, self.id)

        if not self._sidecar_available():
            raise RuntimeError(
                "MLX Z-Image generation requires the isolated mlx-flux sidecar venv "
                "(shared with MlxFluxAdapter / MlxQwenAdapter). The desktop bootstrap "
                "installs it on first launch (apps/desktop/src/setup.rs "
                "provision_mlx_flux_venv); rebuild without SCENEWORKS_DISABLE_MLX_FLUX=1, "
                "or set SCENEWORKS_MLX_FLUX_PYTHON to a Python interpreter that has "
                f"mflux installed (looked for {self._sidecar_python()})."
            )

        steps = self._num_inference_steps(request, model_target)
        guidance = self._guidance_scale(request, model_target)
        quantize = self._resolve_quantize(request)
        if pose_set:
            # One image per pose, sharing a seed so noise-derived attributes stay
            # consistent across the set (mirrors the InstantID / Flux2 pose tiers);
            # only the conditioned body pose changes.
            #
            # Strict ControlNet tier: the skeleton conditions the pose DIRECTLY, so
            # the prompt must stay the plain character prompt. The
            # augment_prompt_for_pose cue ("matching the OpenPose skeleton reference
            # image") is for the best-effort multi-image tier (sc-2256, where the
            # skeleton is a second prompt image) — adding it here makes Z-Image
            # literally draw a skeleton in the scene and fights the pose lock.
            total = len(pose_entries)
            set_seed = resolve_seed(request.seed, request.prompt, 0, request.seeds)
            seeds = [set_seed] * total
            prompts = None
            pose_keypoints = [normalize_keypoints(p.get("keypoints")) for p in pose_entries]
        else:
            total = request.count
            seeds = [resolve_seed(request.seed, request.prompt, index, request.seeds) for index in range(total)]
            prompts = None
            pose_keypoints = None

        progress(
            "loading_model",
            "loading_model",
            0.18,
            f"Loading {model_target['label']} (MLX sidecar venv).",
        )
        work_dir = Path(tempfile.mkdtemp(prefix="mlx_z_image_sidecar_"))
        self._scratch_dir = work_dir
        # Strict pose tier: render each pose's COCO-18 skeleton to a PNG in the
        # sidecar work dir; the runner conditions ZImageControl on it. draw_bodypose
        # runs in the main worker venv (cv2 present).
        control_image_paths: list[str] | None = None
        control_scale: float | None = None
        if pose_set and pose_keypoints is not None:
            control_scale = self._control_scale(request)
            # The Fun-Controlnet-Union pose head is DWPose-trained; a hair-thin
            # skeleton (the default stickwidth=4) at 1024² is a weak, partly
            # out-of-distribution control signal that only steers under a vague
            # prompt. A resolution-proportional stickwidth (~12px at 1024²) gives
            # a strong in-distribution signal that locks the pose cleanly at
            # controlScale 1.0 even under detailed character prompts (sc-2257
            # validation: thin→arms-down, thick→clean T-pose lock).
            stick = max(6, round(min(request.width, request.height) * 0.012))
            control_image_paths = []
            for index, keypoints in enumerate(pose_keypoints):
                # The Fun-Controlnet-Union pose head is trained on full DWPose
                # (body + hands + face). When a pose entry carries hand/face
                # keypoints, render them too (more in-distribution → firmer lock);
                # body-only entries render exactly as before.
                entry = pose_entries[index]
                hands = normalize_hands(entry.get("hands"))
                face = normalize_face(entry.get("face"))
                skeleton_path = work_dir / f"pose_skeleton_{index:04d}.png"
                Image.fromarray(
                    draw_wholebody(request.width, request.height, keypoints, hands=hands, face=face, stickwidth=stick)
                ).save(skeleton_path, "PNG")
                control_image_paths.append(str(skeleton_path))
        try:
            images = self._run_sidecar(
                job_id=job["id"],
                work_dir=work_dir,
                label=model_target["label"],
                total=total,
                spec={
                    "model": request.model,
                    "prompt": request.prompt,
                    "negativePrompt": request.negative_prompt or None,
                    "seeds": seeds,
                    # Per-iteration pose-augmented prompts (None → top-level prompt).
                    "prompts": prompts,
                    "height": request.height,
                    "width": request.width,
                    "numInferenceSteps": steps,
                    "guidance": guidance,
                    "quantize": quantize,
                    "loras": [
                        {"path": lora.path, "weight": lora.weight, "name": lora.adapter_name}
                        for lora in lora_specs
                    ],
                    # Strict pose ControlNet (sc-2257): per-iteration skeletons +
                    # lock strength. None → the plain text-to-image path.
                    "controlImagePaths": control_image_paths,
                    "controlScale": control_scale,
                },
                progress=progress,
                cancel_requested=cancel_requested,
            )

            def image_at_index(index: int) -> Image.Image:
                progress(
                    "running",
                    "generating",
                    image_batch_progress(index, total),
                    format_batch_running_message(model_target["label"], index, total),
                )
                with Image.open(images[index]) as handle:
                    return handle.convert("RGB")

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
                    "repo": model_target["repo"],
                    "numInferenceSteps": steps,
                    "guidanceScale": guidance,
                    "mlxQuantize": quantize,
                    "sidecarVenv": self._sidecar_python(),
                    "realModelInference": True,
                    **({"poseLibrary": True} if pose_set else {}),
                },
                settings=settings,
                job_id=job["id"],
            )
        finally:
            self.discard_temp_outputs(job["id"])

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _control_scale(self, request: ImageRequest) -> float:
        """Pose ControlNet lock strength (sc-2257). Fun-Controlnet-Union recommends
        0.65–1.0; default 0.9 matches the VideoX-Fun reference's own default and
        locks cleanly. Overridable via advanced.controlScale, clamped to [0, 2]."""
        try:
            value = float(request.advanced.get("controlScale", 0.9))
        except (TypeError, ValueError):
            return 0.9
        return max(0.0, min(2.0, value))

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        # Z-Image-Turbo is guidance-distilled; mflux accepts guidance=None to
        # skip the CFG path. Mirror the per-model default from MODEL_TARGETS
        # (which the torch adapter also reads).
        default = model_target.get("guidanceScale", 1.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _resolve_quantize(self, request: ImageRequest) -> int | None:
        """Pick the mflux quantize level. Order: advanced > manifest > Q8 default.
        Identical to MlxFluxAdapter._resolve_quantize / MlxQwenAdapter._resolve_quantize."""
        override = request.advanced.get("mlxQuantize")
        if override is not None and not isinstance(override, bool):
            try:
                parsed = int(override)
                return parsed if parsed > 0 else None
            except (TypeError, ValueError):
                pass
        mlx_entry = request.model_manifest_entry.get("mlx") if request.model_manifest_entry else None
        if isinstance(mlx_entry, dict):
            manifest_q = mlx_entry.get("quantize")
            if manifest_q is not None:
                try:
                    parsed = int(manifest_q)
                    return parsed if parsed > 0 else None
                except (TypeError, ValueError):
                    pass
        return 8

    def _run_sidecar(
        self,
        *,
        job_id: str,
        work_dir: Path,
        label: str,
        total: int,
        spec: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> list[str]:
        spec = {**spec, "outDir": str(work_dir)}
        spec_path = work_dir / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        stdout_log = work_dir / "stdout.log"
        cmd = [self._sidecar_python(), str(self._runner_path()), str(spec_path)]
        emit_worker_event(
            "mlx_z_image_sidecar_start",
            jobId=job_id,
            adapter=self.id,
            model=spec["model"],
            imageCount=total,
            quantize=spec.get("quantize"),
            sidecar=self._sidecar_python(),
        )
        progress(
            "running",
            "generating",
            image_batch_progress(0, total),
            f"Running {label} ({total} image(s)).",
        )
        with stdout_log.open("w", encoding="utf-8") as out:
            # stderr merged into stdout.log so a native crash (SIGABRT/SIGSEGV)
            # leaves a partial traceback we can surface; result.json stays the
            # authoritative success channel (_read_result prefers it).
            proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT)
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_requested():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise InterruptedError("Image generation canceled by user.")
        result = self._read_result(work_dir, stdout_log)
        if proc.returncode != 0 or "error" in result:
            error = result.get("error") or f"MLX Z-Image sidecar exited with code {proc.returncode}."
            error = _mlx_sidecar_failure_detail(error, proc.returncode, stdout_log)
            emit_worker_event(
                "mlx_z_image_sidecar_failed",
                jobId=job_id,
                adapter=self.id,
                error=error,
                returnCode=proc.returncode,
            )
            raise RuntimeError(f"MLX Z-Image generation failed in the sidecar venv: {error}")
        images = [str(path) for path in result.get("images", [])]
        if len(images) != total:
            raise RuntimeError(f"MLX Z-Image sidecar produced {len(images)} image(s); expected {total}.")
        emit_worker_event("mlx_z_image_sidecar_complete", jobId=job_id, adapter=self.id, imageCount=len(images))
        return images

    @staticmethod
    def _read_result(work_dir: Path, stdout_log: Path) -> dict[str, Any]:
        result_path = work_dir / "result.json"
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
        return {"error": "MLX Z-Image sidecar produced no parseable result."}


class MlxSdxlAdapter:
    """Stable Diffusion XL base 1.0 via Apple's mlx-examples (vendored), IN-PROCESS.

    Distinct from the mflux sidecar family (MlxFluxAdapter / MlxQwenAdapter /
    MlxZImageAdapter): mlx-examples ships with the same minimal-dep stack the
    main worker venv already has (mlx, huggingface_hub, regex, numpy, tqdm,
    Pillow) once `requirements-mlx.txt` is installed on macOS — no separate
    venv, no subprocess, no transformers/huggingface_hub conflict like mflux.
    The vendored copy lives at `scene_worker/_vendor/mlx_sd/` (sc-1975, path C).

    Spike measurement (sc-1975, M5 Max 128 GB, SDXL base 1.0, 1024² 30 steps,
    CFG 7, seed 42, bf16): 1.08 s/step (~33 s gen), ~50 GB peak — roughly 3×
    per-step speed vs the torch `SdxlDiffusersAdapter` baseline.

    LoRA support: SceneWorks-authored merge module at
    `_vendor/mlx_sd/lora.py`. Handles **PEFT** (the format SceneWorks's own
    SDXL LoRA training kernel emits — `_SdxlLoraBackend`, training_adapters.py)
    and **kohya with diffusers-style block paths** (the format almost every HF
    community SDXL LoRA ships in, incl. LCM-LoRA). NOT yet supported: original
    SD `input_blocks_*` paths (older offset-LoRA-style LoRAs) and FF-net /
    conv-only LoRAs (mlx-examples renames the GEGLU FF differently than
    diffusers; documented gap, follow-up). The merge is destructive (writes
    into ``unet.<...>.weight`` directly), so the adapter reloads the model
    whenever the LoRA composition changes.

    v1 scope: T2I only on `sdxl`. ``edit_image`` would need a parallel
    img2img variant (mlx-examples ships `image2image.py` but we don't vendor
    it — separate scope). ``reference_asset_id`` jobs (IP-Adapter, sc-2007)
    aren't supported here at all — fall back to the torch path.
    """

    id = "mlx_sdxl"
    _supported_models = {"sdxl"}

    def __init__(self) -> None:
        self._sd: Any | None = None
        # Cache key: (model_repo, frozenset of (lora_path, weight) tuples) — a
        # change in either forces a reload because LoRA merge is destructive.
        self._loaded_key: tuple | None = None

    def loaded_models(self) -> list[str]:
        return [self._loaded_key[0]] if self._loaded_key else []

    def unload(self) -> bool:
        if self._sd is None:
            return False
        self._sd = None
        self._loaded_key = None
        try:
            import mlx.core as mx  # noqa: F401  (lazy — keeps non-Mac hosts importable)
            mx.clear_cache()
        except Exception:
            pass
        return True

    @staticmethod
    def _mlx_sd_available() -> bool:
        # The vendored package only imports cleanly when `mlx` is present in
        # the venv (Apple Silicon + requirements-mlx.txt installed). On
        # Windows / Linux / Docker the module-level `import mlx.core` raises
        # at import time, so existence-check it before claiming we can run.
        try:
            import mlx.core  # noqa: F401
        except Exception:
            return False
        try:
            from . import _vendor  # noqa: F401  (package marker)
            from ._vendor import mlx_sd  # noqa: F401
            return True
        except Exception:
            return False

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
        if model_target.get("adapter") != SdxlDiffusersAdapter.id:
            raise RuntimeError(f"{request.model} is not an SDXL target.")
        if request.model not in self._supported_models:
            raise RuntimeError(
                f"MlxSdxlAdapter supports "
                f"{', '.join(sorted(self._supported_models))}, not {request.model}."
            )
        if request.mode == "edit_image":
            raise RuntimeError(
                f"{request.model} MLX adapter is text-to-image only (v1). "
                "Use SCENEWORKS_IMAGE_ADAPTER=sdxl_diffusers for the torch edit path."
            )
        if request.reference_asset_id:
            raise RuntimeError(
                f"{request.model} reference-image (IP-Adapter) generation is not "
                "supported on the MLX backend. Use the torch path "
                "(SCENEWORKS_IMAGE_ADAPTER=sdxl_diffusers)."
            )
        if not self._mlx_sd_available():
            raise RuntimeError(
                "MLX SDXL generation requires the macOS-only MLX install "
                "(apps/worker/requirements-mlx.txt) — looked for `mlx.core` and "
                "the vendored `_vendor/mlx_sd` package, one or both not "
                "importable in this worker venv."
            )

        validate_lora_compatibility(
            request.loras, model_family=model_target.get("family"), adapter_id=self.id, model_id=request.model
        )
        lora_specs = normalize_lora_specs(request.loras)
        # The MLX backend can't apply LoKr (its merge math is LoRA-only); the SDXL
        # family can produce LoKr adapters, so reject them clearly here (epic 2193).
        reject_lokr_loras(lora_specs, self.id)

        total = request.count
        steps = self._num_inference_steps(request, model_target)
        guidance = self._guidance_scale(request, model_target)
        seeds = [resolve_seed(request.seed, request.prompt, index, request.seeds) for index in range(total)]
        repo = self._repo_for_request(request, model_target)

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']} (MLX).")
        sd = self._load_sd(repo, lora_specs, job_id=job["id"])

        # mlx-examples treats `latent_size` as the post-VAE downscale (×8).
        latent_h, latent_w = request.height // 8, request.width // 8

        def image_at_index(index: int) -> Image.Image:
            seed = seeds[index]
            progress(
                "running",
                "generating",
                image_batch_progress(index, total),
                format_batch_running_message(model_target["label"], index, total),
            )
            emit_worker_event(
                "image_inference_start",
                jobId=job["id"],
                adapter=self.id,
                model=request.model,
                imageIndex=index,
                imageCount=total,
                device="mps",
            )
            try:
                pil = self._run_one(sd, request.prompt, request.negative_prompt or "", seed, steps, guidance, latent_h, latent_w)
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
            )
            return pil

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
                "repo": repo,
                "numInferenceSteps": steps,
                "guidanceScale": guidance,
                "mlxBackend": "mlx-examples/stable_diffusion (vendored)",
                "realModelInference": True,
            },
            settings=settings,
            job_id=job["id"],
        )

    def _load_sd(
        self,
        repo: str,
        lora_specs: list[Any],
        *,
        job_id: str,
    ) -> Any:
        # Lazy imports — the module-level safety check is in _mlx_sd_available;
        # by the time we reach here mlx + the vendored package must import.
        from ._vendor.mlx_sd import StableDiffusionXL  # type: ignore[import-not-found]
        from ._vendor.mlx_sd.lora import apply_loras_to_unet  # type: ignore[import-not-found]

        lora_key = tuple(sorted((spec.path, float(spec.weight)) for spec in lora_specs))
        cache_key = (repo, lora_key)
        if self._sd is not None and self._loaded_key == cache_key:
            return self._sd

        emit_worker_event(
            "mlx_sdxl_load_start",
            jobId=job_id,
            adapter=self.id,
            repo=repo,
            loraCount=len(lora_specs),
        )
        # Clear the previous resident UNet before loading a new one — saves
        # peak memory on small Macs that just barely fit one SDXL UNet at a
        # time. mlx_sd doesn't expose a `del unet` hook but Python's GC + the
        # MLX cache clear handles it.
        if self._sd is not None:
            self._sd = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

        sd = StableDiffusionXL(repo, float16=True)
        if lora_specs:
            specs_payload = [
                {"path": spec.path, "weight": float(spec.weight)} for spec in lora_specs
            ]
            touched = apply_loras_to_unet(sd.unet, specs_payload)
            if touched == 0:
                # All LoRAs in unsupported formats / no matching modules — surface
                # rather than silently shipping an unmodified model.
                raise RuntimeError(
                    "MLX SDXL LoRA merge found no matching modules for the supplied "
                    "LoRAs. Supported formats: PEFT (lora_A/lora_B) and kohya with "
                    "diffusers-style block paths (lora_unet_down_blocks_..., "
                    "lora_unet_up_blocks_..., lora_unet_mid_block_...). Original-SD "
                    "input_blocks paths and FF-net/conv-only LoRAs aren't merged in v1."
                )
            emit_worker_event(
                "mlx_sdxl_lora_merged",
                jobId=job_id,
                adapter=self.id,
                loraCount=len(lora_specs),
                modulesTouched=touched,
            )
        sd.ensure_models_are_loaded()
        self._sd = sd
        self._loaded_key = cache_key
        emit_worker_event(
            "mlx_sdxl_load_complete",
            jobId=job_id,
            adapter=self.id,
            repo=repo,
        )
        return sd

    @staticmethod
    def _run_one(
        sd: Any,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        latent_h: int,
        latent_w: int,
    ) -> Image.Image:
        import mlx.core as mx
        import numpy as np

        latents = sd.generate_latents(
            prompt,
            n_images=1,
            cfg_weight=guidance,
            num_steps=steps,
            seed=seed,
            negative_text=negative_prompt,
            latent_size=(latent_h, latent_w),
        )
        x_t = None
        for x_t in latents:
            mx.eval(x_t)
        # Decode + scale [0, 1] → uint8 (mlx-examples txt2image.py recipe).
        image = sd.decode(x_t)
        mx.eval(image)
        image = (image * 255).astype(mx.uint8)
        # `image[0]` drops the batch dim; np.array works directly on the
        # mx.array buffer (no tolist() — that collapses dtype).
        return Image.fromarray(np.array(image[0])).convert("RGB")

    def _repo_for_request(self, request: ImageRequest, model_target: dict[str, Any]) -> str:
        return request.advanced.get("modelRepo") or model_target["repo"]

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target.get("steps", 30), 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = model_target.get("guidanceScale", 7.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)


class MlxFlux2Adapter:
    """FLUX.2-klein-9b (txt2img + edit) via mflux (Apple MLX), run OUT-OF-PROCESS.

    The first MLX-only image-generation family in SceneWorks — there is no
    diffusers torch fallback for FLUX.2 today, so the model's manifest entry
    points directly at this adapter rather than redirecting from a torch
    adapter. On non-Mac hosts the dispatch falls through to nothing and the
    job fails with a clear error; the frontend hides the model on those
    platforms via the existing host-capability filter.

    Shares the mflux sidecar venv (`/opt/mlx-flux-venv`) and runner
    (`scene_worker/mlx_flux_runner.py`) with `MlxFluxAdapter` /
    `MlxQwenAdapter` / `MlxZImageAdapter` — runner dispatches on
    `spec["model"]` to `Flux2Klein` (txt2img) or `Flux2KleinEdit` (reference).

    Two model ids:
      - ``flux2_klein_9b`` — txt2img + edit. Edit mode passes a single
        reference image through `image_paths` and runs the standard
        denoise loop.
      - ``flux2_klein_9b_kv`` — edit-only. The KV-cache distilled variant
        from BFL (FLUX.2-klein-9b-kv) caches reference-image K/V on step
        0 and reuses it on steps 1-3, ~2.4× faster than the un-cached
        edit path on M5 Max. Cache auto-engages via the mflux
        ModelConfig flag (sc-2163, upstream PR filipstrand/mflux#426). The
        cache only engages on the edit path when a reference is present;
        without one the id runs plain txt2img through ``Flux2Klein`` just
        like the base 9B (sc-2173).

    Dispatch gate: `_should_route_flux2_to_mlx`. The shared mflux escape
    hatch ``SCENEWORKS_DISABLE_MLX_FLUX`` opts out of this adapter too.

    Validation (M5 Max 128GB, 1024², 4 steps):
      - flux2_klein_9b txt2img: ~26 s gen, ~36 GB peak
      - flux2_klein_9b edit (no cache): ~33 s gen
      - flux2_klein_9b_kv edit (cache on): ~13.5 s gen — 2.4× speedup
      - flux2_klein_9b_kv txt2img: parity with base 9B, no cache artifacts
        (sc-2173)
      (numbers measured against the mflux fork's editable install during
       sc-2163/sc-2173; sidecar venv post-provision should match).
    """

    id = "mlx_flux2"
    _supported_models = {"flux2_klein_9b", "flux2_klein_9b_kv", "flux2_klein_9b_true_v2"}
    # Models whose weights load from a locally-assembled diffusers dir (produced
    # by the install-time conversion job, sc-2235) rather than an mflux built-in
    # repo. The adapter resolves the dir and threads it to the runner as modelPath.
    _local_dir_models = {"flux2_klein_9b_true_v2"}
    # Models whose edit path uses the KV cache (engages only with a reference);
    # they still run plain txt2img without one (sc-2173).
    _kv_cache_models = {"flux2_klein_9b_kv"}

    def __init__(self) -> None:
        self._scratch_dir: Path | None = None

    @classmethod
    def _local_model_dir(cls, model_id: str, settings: WorkerSettings) -> str | None:
        """Local assembled diffusers dir for converted fine-tunes (sc-2235), or
        None for built-in mflux repos. Convention mirrors the model_convert job's
        output dir (``<data_dir>/models/mlx/<model_id>``). Returns None if not yet
        converted so the runner surfaces a clear missing-weights error."""
        if model_id not in cls._local_dir_models:
            return None
        candidate = Path(settings.data_dir) / "models" / "mlx" / model_id
        return str(candidate) if (candidate / "transformer").is_dir() else None

    def discard_temp_outputs(self, job_id: str | None = None) -> None:
        work_dir = self._scratch_dir
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._scratch_dir = None

    def loaded_models(self) -> list[str]:
        # The sidecar process loads + frees the model per job; nothing stays
        # resident in this (main-venv) process. Mirrors MlxFluxAdapter.
        return []

    @staticmethod
    def _sidecar_python() -> str:
        return os.getenv("SCENEWORKS_MLX_FLUX_PYTHON", "/opt/mlx-flux-venv/bin/python")

    @staticmethod
    def _runner_path() -> Path:
        return Path(__file__).resolve().parent / "mlx_flux_runner.py"

    def _sidecar_available(self) -> bool:
        return Path(self._sidecar_python()).exists() and self._runner_path().exists()

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
            raise RuntimeError(f"{request.model} is not a FLUX.2 target.")
        if request.model not in self._supported_models:
            raise RuntimeError(
                f"MlxFlux2Adapter supports "
                f"{', '.join(sorted(self._supported_models))}, not {request.model}."
            )

        # Resolve the reference image (if any) to a local filesystem path so
        # mflux's Flux2KleinEdit can read it directly. Two studios feed the same
        # single-reference edit path: Character Studio sends it as
        # referenceAssetId, while the Image Edit studio sends the source image as
        # sourceAssetId (edit_image mode). No PIL round-trip: find_asset_media_path
        # returns the project-scoped path and mflux's loader handles dtype/resize.
        reference_paths: list[str] = []
        edit_source_id = request.source_asset_id if request.mode == "edit_image" else None
        ref_id = request.reference_asset_id or edit_source_id
        if ref_id:
            reference_paths.append(str(find_asset_media_path(project_path, ref_id)))
        has_reference = bool(reference_paths)

        validate_lora_compatibility(
            request.loras, model_family=model_target.get("family"), adapter_id=self.id, model_id=request.model
        )
        lora_specs = normalize_lora_specs(request.loras)
        # The MLX backend can't apply LoKr (its merge math is LoRA-only); reject
        # clearly rather than silently ignoring the adapter (epic 2193).
        reject_lokr_loras(lora_specs, self.id)

        if not self._sidecar_available():
            raise RuntimeError(
                "MLX FLUX.2 generation requires the isolated mlx-flux sidecar venv. The desktop "
                "bootstrap installs it on first launch (apps/desktop/src/setup.rs "
                "provision_mlx_flux_venv); rebuild without SCENEWORKS_DISABLE_MLX_FLUX=1, or set "
                "SCENEWORKS_MLX_FLUX_PYTHON to a Python interpreter that has mflux installed "
                f"(looked for {self._sidecar_python()})."
            )

        # sc-2003 multi-backbone angle set: when advanced.angleSet is set on a
        # character_image request with a reference, loop the 11 canonical
        # angles in one job — same shape as InstantID's angle set (sc-2050),
        # but the per-angle prompt augment comes from
        # character_studio_angles.ANGLE_PROMPT_AUGMENTS (Flux2 has no landmark
        # ControlNet; the angle comes from prompt-driven editing of the
        # reference). Spike-validated: FLUX.2-klein mean ArcFace 0.52 across
        # angles AND uniquely holds portrait framing at 90° profiles where
        # Qwen-Lightning reframes to full-body.
        is_character_image = request.mode == "character_image" and has_reference
        # Best-effort pose tier (sc-2262): advanced.poses is a list of {id, keypoints}.
        # Each pose renders to an OpenPose skeleton paired with the reference as a
        # [skeleton, reference] multi-image set (Flux2KleinEdit accepts a list of
        # image_paths) — identity from the reference, pose approximated from the
        # skeleton + the pose prompt cue. No FLUX.2 pose ControlNet exists (sc-2250).
        # Pose takes precedence over angleSet.
        raw_poses = request.advanced.get("poses")
        pose_entries = [p for p in raw_poses if isinstance(p, dict)] if isinstance(raw_poses, list) else []
        pose_set = is_character_image and len(pose_entries) > 0
        angle_set = is_character_image and bool(request.advanced.get("angleSet")) and not pose_set
        pose_keypoints: list[Any] | None = None
        if pose_set:
            total = len(pose_entries)
            angles = None
            set_seed = resolve_seed(request.seed, request.prompt, 0, request.seeds)
            seeds = [set_seed] * total
            prompts = [augment_prompt_for_pose(request.prompt)] * total
            pose_keypoints = [normalize_keypoints(p.get("keypoints")) for p in pose_entries]
        elif angle_set:
            angles = list(CHARACTER_ANGLE_SET_ORDER)
            total = len(angles)
            set_seed = resolve_seed(request.seed, request.prompt, 0, request.seeds)
            # Shared seed across the set so noise-derived attributes (hair,
            # lighting) stay consistent across angles — only the head pose
            # changes. Mirrors the sc-2050 InstantID angle-set seed strategy.
            seeds = [set_seed] * total
            prompts = [augment_prompt_for_angle(request.prompt, angle) for angle in angles]
        else:
            total = request.count
            angles = None
            seeds = [resolve_seed(request.seed, request.prompt, index, request.seeds) for index in range(total)]
            prompts = None
        steps = self._num_inference_steps(request, model_target)
        guidance = self._guidance_scale(request, model_target)
        quantize = self._resolve_quantize(request)

        progress(
            "loading_model",
            "loading_model",
            0.18,
            f"Loading {model_target['label']} (MLX sidecar venv).",
        )
        work_dir = Path(tempfile.mkdtemp(prefix="mlx_flux2_sidecar_"))
        self._scratch_dir = work_dir
        # Best-effort pose tier: render each pose's skeleton to a PNG in the sidecar
        # work dir and pair it with the reference as a per-iteration [reference,
        # skeleton] set. draw_bodypose runs in the main worker venv (cv2 present).
        image_paths_per_iter: list[list[str]] | None = None
        if pose_set and pose_keypoints is not None:
            image_paths_per_iter = []
            for index, keypoints in enumerate(pose_keypoints):
                skeleton_path = work_dir / f"pose_skeleton_{index:04d}.png"
                Image.fromarray(draw_bodypose(request.width, request.height, keypoints)).save(
                    skeleton_path, "PNG"
                )
                # Order [skeleton, reference] mirrors the sc-2003 spike's validated
                # FLUX.2 multi-image config (image_paths=[skeleton, character]).
                image_paths_per_iter.append([str(skeleton_path), reference_paths[0]])
        try:
            images = self._run_sidecar(
                job_id=job["id"],
                work_dir=work_dir,
                label=model_target["label"],
                total=total,
                spec={
                    "model": request.model,
                    "prompt": request.prompt,
                    "negativePrompt": None,  # Flux2 disallows negatives; runner skips it anyway
                    "seeds": seeds,
                    # Per-iteration prompt overrides — runner zips with seeds.
                    # None / absent → all iterations use the top-level "prompt".
                    "prompts": prompts,
                    "height": request.height,
                    "width": request.width,
                    "numInferenceSteps": steps,
                    "guidance": guidance,
                    "quantize": quantize,
                    "loras": [
                        {"path": lora.path, "weight": lora.weight, "name": lora.adapter_name}
                        for lora in lora_specs
                    ],
                    "imagePaths": reference_paths or None,
                    # Per-pose [reference, skeleton] sets (sc-2262); overrides
                    # imagePaths per iteration. None for the plain reference path.
                    "imagePathsPerIteration": image_paths_per_iter,
                    # Local diffusers dir for converted fine-tunes (sc-2235);
                    # None for built-in mflux repos (loaded from ModelConfig).
                    "modelPath": self._local_model_dir(request.model, settings),
                },
                progress=progress,
                cancel_requested=cancel_requested,
            )

            def image_at_index(index: int) -> Image.Image:
                progress(
                    "running",
                    "generating",
                    image_batch_progress(index, total),
                    format_batch_running_message(model_target["label"], index, total),
                )
                with Image.open(images[index]) as handle:
                    return handle.convert("RGB")

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
                    "repo": model_target["repo"],
                    "numInferenceSteps": steps,
                    "guidanceScale": guidance,
                    "mlxQuantize": quantize,
                    "sidecarVenv": self._sidecar_python(),
                    "hasReference": has_reference,
                    "kvCacheEnabled": has_reference and request.model in self._kv_cache_models,
                    "realModelInference": True,
                    **({"angleSet": True} if angle_set else {}),
                    **({"poseLibrary": True} if pose_set else {}),
                },
                settings=settings,
                job_id=job["id"],
            )
        finally:
            self.discard_temp_outputs(job["id"])

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        # FLUX.2-klein distilled variants are 4-step; cap at 40 because the
        # ModelConfig has supports_guidance=True so power users CAN crank
        # steps if they really want.
        return safe_int(request.advanced.get("steps"), model_target.get("steps", 4), 1, 40)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        # Distilled FLUX.2 wants guidance=1.0 (the mflux CLI even errors if
        # not 1.0 on distilled variants). Leave room for manifest override
        # in case a future base variant needs a different default.
        default = model_target.get("guidanceScale", 1.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _resolve_quantize(self, request: ImageRequest) -> int | None:
        """Pick the mflux quantize level. Same algorithm as MlxFluxAdapter:
        advanced override > manifest mlx.quantize > Q8 default.

        FLUX.2-klein-9b at bf16 is ~36GB peak on M5 Max; Q8 cuts that
        substantially with no visible quality drift on the spike runs.
        """
        override = request.advanced.get("mlxQuantize")
        if override is not None and not isinstance(override, bool):
            try:
                parsed = int(override)
                return parsed if parsed > 0 else None
            except (TypeError, ValueError):
                pass
        mlx_entry = request.model_manifest_entry.get("mlx") if request.model_manifest_entry else None
        if isinstance(mlx_entry, dict):
            manifest_q = mlx_entry.get("quantize")
            if manifest_q is not None:
                try:
                    parsed = int(manifest_q)
                    return parsed if parsed > 0 else None
                except (TypeError, ValueError):
                    pass
        return 8

    def _run_sidecar(
        self,
        *,
        job_id: str,
        work_dir: Path,
        label: str,
        total: int,
        spec: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> list[str]:
        spec = {**spec, "outDir": str(work_dir)}
        spec_path = work_dir / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        stdout_log = work_dir / "stdout.log"
        cmd = [self._sidecar_python(), str(self._runner_path()), str(spec_path)]
        emit_worker_event(
            "mlx_flux2_sidecar_start",
            jobId=job_id,
            adapter=self.id,
            model=spec["model"],
            imageCount=total,
            quantize=spec.get("quantize"),
            sidecar=self._sidecar_python(),
            hasReference=bool(spec.get("imagePaths")),
        )
        progress(
            "running",
            "generating",
            image_batch_progress(0, total),
            f"Running {label} ({total} image(s)).",
        )
        with stdout_log.open("w", encoding="utf-8") as out:
            # stderr merged into stdout.log so a native crash (SIGABRT/SIGSEGV)
            # leaves a partial traceback we can surface; result.json stays the
            # authoritative success channel (_read_result prefers it).
            proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT)
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_requested():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise InterruptedError("Image generation canceled by user.")
        result = MlxFluxAdapter._read_result(work_dir, stdout_log)
        if proc.returncode != 0 or "error" in result:
            error = result.get("error") or f"MLX FLUX.2 sidecar exited with code {proc.returncode}."
            error = _mlx_sidecar_failure_detail(error, proc.returncode, stdout_log)
            emit_worker_event(
                "mlx_flux2_sidecar_failed",
                jobId=job_id,
                adapter=self.id,
                error=error,
                returnCode=proc.returncode,
            )
            raise RuntimeError(f"MLX FLUX.2 generation failed in the sidecar venv: {error}")
        images = [str(path) for path in result.get("images", [])]
        if len(images) != total:
            raise RuntimeError(f"MLX FLUX.2 sidecar produced {len(images)} image(s); expected {total}.")
        emit_worker_event("mlx_flux2_sidecar_complete", jobId=job_id, adapter=self.id, imageCount=len(images))
        return images


class KolorsDiffusersAdapter:
    """Kwai-Kolors Kolors text-to-image via diffusers.KolorsPipeline.

    Mirrors FluxDiffusersAdapter: HF-cache check, device/dtype selection, progress
    + cancel callbacks, incremental asset writing, and worker events for pipeline
    load + inference. Runs in the MAIN worker venv (diffusers >= 0.30 + transformers
    + the bundled ChatGLM3 text encoder) — no sidecar.

    Unlike FLUX, Kolors uses real classifier-free guidance, so it honors the
    negative prompt and a non-zero guidance_scale (~5.0). The Kolors-diffusers
    checkpoint ships EulerDiscreteScheduler; we switch to DPMSolverMultistep with
    Karras sigmas per the model card for the recommended ~25-step config.

    Supports text-to-image (KolorsPipeline) and img2img editing
    (KolorsImg2ImgPipeline) on the same checkpoint, switched by request.mode.
    """

    id = "kolors_diffusers"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._img2img_pipe: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_model: str | None = None
        # Whether the resident text pipe has the IP-Adapter (+ image encoder) loaded.
        # A plain-T2I pipe and an IP-Adapter pipe are not interchangeable, so this is
        # part of the text-pipe cache key.
        self._text_ip_adapter: bool = False
        # Strict pose tier (sc-2264): a separate vendored ControlNet img2img pipeline
        # (pose ControlNet + IP-Adapter identity). Kept apart from the T2I/img2img pipes
        # since it carries the extra ControlNet weights.
        self._pose_pipe: Any | None = None
        self._pose_loaded_repo: str | None = None
        self._loaded_lora_states: dict[str, LoraPipelineState] = {}

    @staticmethod
    def _use_ip_adapter(request: ImageRequest) -> bool:
        # IP-Adapter reference conditioning runs on the text-to-image pipeline only
        # (reference + img2img edit together is a future enhancement).
        return request.mode != "edit_image" and bool(request.reference_asset_id)

    @staticmethod
    def _pose_entries(request: ImageRequest) -> list[dict[str, Any]]:
        """Pose-library entries ({id, keypoints}) when this is a character_image pose
        job with a reference. Empty otherwise (no pose ControlNet path)."""
        if request.mode == "edit_image" or not request.reference_asset_id:
            return []
        raw = request.advanced.get("poses")
        return [p for p in raw if isinstance(p, dict)] if isinstance(raw, list) else []

    def _openpose_scale(self, request: ImageRequest) -> float:
        try:
            return max(0.0, min(2.0, float(request.advanced.get("openPoseScale", 0.7))))
        except (TypeError, ValueError):
            return 0.7

    def _generate_pose_set(
        self,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
        model_target: dict[str, Any],
        pose_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Strict pose tier (sc-2264): one image per library pose via the vendored
        Kolors pose ControlNet pipeline (pose ControlNet structure + IP-Adapter
        identity), sharing one seed so wardrobe/hair stay consistent across the set."""
        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']} (pose ControlNet).")
        pipe = self._load_pose_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        total = len(pose_entries)
        pose_keypoints = [normalize_keypoints(p.get("keypoints")) for p in pose_entries]
        set_seed = resolve_seed(request.seed, request.prompt, 0, request.seeds)
        label = f"{model_target['label']} pose"

        def image_at_index(index: int) -> Image.Image:
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
                poseId=pose_entries[index].get("id"),
                device=device,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            try:
                image = self._run_pose(
                    settings, pipe, request, set_seed, project_path, pose_keypoints[index],
                    cancel_requested=cancel_requested,
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
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
                "repo": self._repo_for_request(request, model_target),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request, model_target),
                "controlNetPose": model_target["controlNetPose"]["repo"],
                "openPoseScale": self._openpose_scale(request),
                "ipAdapterScale": self._ip_adapter_scale(request),
                "poseLibrary": True,
                "realModelInference": True,
            },
        )

    def _load_pose_pipeline(
        self,
        settings: WorkerSettings,
        request: ImageRequest,
        model_target: dict[str, Any],
        progress: ProgressCallback,
        *,
        job_id: str,
    ) -> Any:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        transformers = importlib.import_module("transformers")
        repo = self._repo_for_request(request, model_target)
        cn = model_target.get("controlNetPose") or {}
        ip_adapter = model_target.get("ipAdapter") or {}
        if not cn or not ip_adapter:
            raise RuntimeError(f"{request.model} has no Kolors pose ControlNet / IP-Adapter configuration.")
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        if self._pose_pipe is not None and self._pose_loaded_repo == repo:
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']} pose pipeline.")
            return self._pose_pipe
        # Only one ~24GB+ pipeline resident at a time: drop the T2I/img2img pipes (and
        # any stale pose pipe) before loading the ControlNet + IP-Adapter stack.
        self._evict_pipelines(torch)
        # Vendored Kolors ControlNet model + img2img pipeline (sc-2264). The pipeline
        # drives diffusers' UNet2DConditionModel, so it composes with the same Kolors
        # components diffusers.KolorsPipeline loads.
        from ._vendor.kolors.models.controlnet import ControlNetModel as KolorsControlNetModel
        from ._vendor.kolors.pipelines.pipeline_controlnet_xl_kolors_img2img import (
            StableDiffusionXLControlNetImg2ImgPipeline as KolorsControlNetPipeline,
        )

        cache_action = "Loading cached" if huggingface_repo_cache_exists(repo) else "Downloading"
        progress("loading_model", "loading_model", 0.2, f"{cache_action} {model_target['label']} + pose ControlNet.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            controlNetPose=cn["repo"],
            useIpAdapter=True,
        )
        base = diffusers.KolorsPipeline.from_pretrained(
            repo, torch_dtype=dtype, variant=model_target.get("variant")
        )
        controlnet = KolorsControlNetModel.from_pretrained(
            cn["repo"], revision=cn.get("revision"), torch_dtype=dtype
        )
        encoder_class = getattr(transformers, "CLIPVisionModelWithProjection", None)
        if encoder_class is None:
            raise RuntimeError(
                "transformers does not expose CLIPVisionModelWithProjection, "
                "required for the Kolors IP-Adapter image encoder."
            )
        image_encoder = encoder_class.from_pretrained(
            ip_adapter["repo"],
            subfolder="image_encoder",
            revision=ip_adapter.get("revision"),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        feature_extractor = transformers.CLIPImageProcessor(size=336, crop_size=336)
        pipe = KolorsControlNetPipeline(
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            unet=base.unet,
            controlnet=controlnet,
            scheduler=base.scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            force_zeros_for_empty_prompt=False,
        )
        pipe.load_ip_adapter(
            ip_adapter["repo"],
            subfolder="",
            weight_name=ip_adapter["weight"],
            revision=ip_adapter.get("revision"),
            image_encoder_folder=None,
        )
        pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        self._pose_pipe = pipe
        self._pose_loaded_repo = repo
        self._loaded_model = request.model
        return pipe

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
        """Render the character in one library pose: OpenPose skeleton drives the pose
        ControlNet, the reference drives identity via IP-Adapter. img2img init is the
        reference (at full strength it only seeds latent dimensions)."""
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        model_target = MODEL_TARGETS[request.model]
        width, height = request.width, request.height
        skeleton = Image.fromarray(draw_bodypose(width, height, keypoints))
        reference = load_reference_image(project_path, request.reference_asset_id)
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        kwargs = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "image": reference,
            "control_image": skeleton,
            "ip_adapter_image": reference,
            "controlnet_conditioning_scale": self._openpose_scale(request),
            "control_guidance_end": 0.9,
            "strength": float(request.advanced.get("strength", 1.0)),
            "height": height,
            "width": width,
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

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._loaded_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        """Free any resident pipeline so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._text_pipe is None and self._img2img_pipe is None and getattr(self, "_pose_pipe", None) is None:
            return False
        self._evict_pipelines(importlib.import_module("torch"))
        return True

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
        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")
        model_target = MODEL_TARGETS.get(request.model, {})
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a Kolors Diffusers target.")

        # Strict pose tier (sc-2264): advanced.poses + a reference routes to the pose
        # ControlNet pipeline (pose ControlNet + IP-Adapter identity), a separate path
        # from the T2I/img2img generate flow below.
        pose_entries = self._pose_entries(request)
        if pose_entries and model_target.get("controlNetPose") and model_target.get("ipAdapter"):
            return self._generate_pose_set(
                settings, job, request, project_path, progress, cancel_requested, model_target, pose_entries
            )

        use_img2img = request.mode == "edit_image"
        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job["id"],
            adapter=self.id,
            loraCount=len(request.loras),
        )
        self._apply_loras(pipe, request)
        emit_worker_event("image_lora_apply_complete", jobId=job["id"], adapter=self.id)
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = f"{model_target['label']} edit" if use_img2img else model_target["label"]

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
                gpuMemory=gpu_memory_snapshot(torch, device),
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
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
                "repo": self._repo_for_request(request, model_target),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request, model_target),
                "maxSequenceLength": self._max_sequence_length(request, model_target),
                "realModelInference": True,
            },
        )

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
        diffusers = importlib.import_module("diffusers")
        repo = self._repo_for_request(request, model_target)
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        use_img2img = request.mode == "edit_image"
        use_ip_adapter = self._use_ip_adapter(request)
        ip_adapter = model_target.get("ipAdapter") or {}
        if use_ip_adapter and not ip_adapter:
            raise RuntimeError(
                f"{request.model} does not support reference-image (IP-Adapter) generation."
            )
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        cached_pipe = self._img2img_pipe if use_img2img else self._text_pipe
        # The text pipe's IP-Adapter state is part of its cache key: a plain-T2I pipe
        # cannot serve a reference job, and vice versa.
        text_state_matches = use_img2img or self._text_ip_adapter == use_ip_adapter
        if cached_pipe is not None and self._loaded_repo == repo and text_state_matches:
            self._loaded_model = request.model
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                model=request.model,
                repo=repo,
                device=device,
                componentDevices=pipeline_component_devices(cached_pipe),
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return cached_pipe

        # Hold only one pipeline per repo: evict stale pipes (other mode, stale repo,
        # or a text pipe whose IP-Adapter state no longer matches) before loading so we
        # don't keep two ~16GB Kolors pipelines resident.
        if self._loaded_repo and self._loaded_repo != repo:
            self._evict_pipelines(torch)
        elif use_img2img:
            if self._text_pipe is not None:
                self._text_pipe = None
                self._text_ip_adapter = False
                self._forget_loaded_loras("text")
                self._empty_cuda_cache(torch)
        else:
            if self._text_pipe is not None:
                self._text_pipe = None
                self._forget_loaded_loras("text")
                self._empty_cuda_cache(torch)
            if self._img2img_pipe is not None:
                self._img2img_pipe = None
                self._forget_loaded_loras("img2img")
                self._empty_cuda_cache(torch)

        pipeline_name = "KolorsImg2ImgPipeline" if use_img2img else "KolorsPipeline"
        pipeline_class = getattr(diffusers, pipeline_name, None)
        if pipeline_class is None:
            raise RuntimeError(
                f"The installed diffusers package does not expose {pipeline_name}. "
                "Install diffusers >= 0.30 for Kolors support."
            )

        cache_action = "Loading cached" if huggingface_repo_cache_exists(repo) else "Downloading"
        progress("loading_model", "loading_model", 0.2, f"{cache_action} {model_target['label']} model files.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useImg2img=use_img2img,
            useIpAdapter=use_ip_adapter,
            cpuOffload=cpu_offload,
            cached=cache_action == "Loading cached",
        )
        # fp16-only repo: pass the variant or from_pretrained looks for the
        # nonexistent default diffusion_pytorch_model.bin and fails (e.g. in vae/).
        from_pretrained_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "variant": model_target.get("variant"),
        }
        if use_ip_adapter:
            # IP-Adapter-Plus needs its own CLIP-ViT-L-336 image encoder, loaded from a
            # PR revision that ships safetensors; pass it in so load_ip_adapter wires it
            # up (with image_encoder_folder=None below).
            transformers = importlib.import_module("transformers")
            encoder_class = getattr(transformers, "CLIPVisionModelWithProjection", None)
            if encoder_class is None:
                raise RuntimeError(
                    "transformers does not expose CLIPVisionModelWithProjection, "
                    "required for the Kolors IP-Adapter image encoder."
                )
            from_pretrained_kwargs["image_encoder"] = encoder_class.from_pretrained(
                ip_adapter["repo"],
                subfolder="image_encoder",
                revision=ip_adapter.get("revision"),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        pipe = pipeline_class.from_pretrained(repo, **from_pretrained_kwargs)
        # Kolors-diffusers ships EulerDiscreteScheduler; the model card recommends
        # DPMSolverMultistep with Karras sigmas for the ~25-step config.
        scheduler_class = getattr(diffusers, "DPMSolverMultistepScheduler", None)
        if scheduler_class is not None and getattr(pipe, "scheduler", None) is not None:
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        if use_ip_adapter:
            # image_encoder_folder=None: we already supplied the encoder above.
            pipe.load_ip_adapter(
                ip_adapter["repo"],
                subfolder="",
                weight_name=ip_adapter["weight"],
                revision=ip_adapter.get("revision"),
                image_encoder_folder=None,
            )
            pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        # IP-Adapter pushes Kolors past ~24GB; if memory is tight enable cpuOffload in
        # advanced settings (forcing offload on MPS has its own caveats, so it stays opt-in).
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        # VAE tiling keeps high-resolution decodes within memory.
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        elif hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        component_devices = verify_pipeline_on_device(
            pipe,
            requested_device=device,
            model_label=model_target["label"],
            allow_offload=offload_enabled,
        )
        emit_worker_event(
            "image_pipeline_on_device",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            requestedDevice=device,
            cpuOffload=offload_enabled,
            componentDevices=component_devices,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        if use_img2img:
            self._img2img_pipe = pipe
        else:
            self._text_pipe = pipe
            self._text_ip_adapter = use_ip_adapter
        self._loaded_repo = repo
        self._loaded_model = request.model
        return pipe

    def _evict_pipelines(self, torch: Any) -> None:
        self._text_pipe = None
        self._img2img_pipe = None
        self._text_ip_adapter = False
        self._pose_pipe = None
        self._pose_loaded_repo = None
        self._loaded_repo = None
        self._loaded_model = None
        self._loaded_lora_states.clear()
        self._empty_cuda_cache(torch)

    def _empty_cuda_cache(self, torch: Any) -> None:
        release_inference_memory(torch)

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
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        model_target = MODEL_TARGETS[request.model]
        kwargs = {
            "prompt": request.prompt,
            # Kolors uses real classifier-free guidance, so the negative prompt
            # is honored (unlike the guidance-distilled FLUX path).
            "negative_prompt": request.negative_prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": self._num_inference_steps(request, model_target),
            "guidance_scale": self._guidance_scale(request, model_target),
            "max_sequence_length": self._max_sequence_length(request, model_target),
            "generator": generator,
        }
        if request.mode == "edit_image":
            # KolorsImg2ImgPipeline blends a source image; strength controls how
            # far the result moves from it (0 = unchanged, 1 = full re-generation).
            kwargs["image"] = load_source_image(project_path, request)
            kwargs["strength"] = float(request.advanced.get("strength", 0.6))
        elif self._use_ip_adapter(request):
            # IP-Adapter conditions T2I on a reference image (style/identity). Vary
            # prompt/seed across the batch to get many images of the same subject.
            kwargs["ip_adapter_image"] = load_reference_image(project_path, request.reference_asset_id)
            if hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**filter_call_kwargs(pipe, kwargs))
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        return output.images[0].convert("RGB")

    def _apply_loras(self, pipe: Any, request: ImageRequest) -> None:
        key = "img2img" if request.mode == "edit_image" else "text"
        model_target = MODEL_TARGETS.get(request.model, {})
        self._loaded_lora_states[key] = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            previous_state=self._loaded_lora_states.get(key),
        )

    def _forget_loaded_loras(self, key: str) -> None:
        self._loaded_lora_states.pop(key, None)

    def _repo_for_request(self, request: ImageRequest, model_target: dict[str, Any]) -> str:
        return request.advanced.get("modelRepo") or model_target["repo"]

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = model_target.get("guidanceScale", 5.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _max_sequence_length(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(
            request.advanced.get("maxSequenceLength"),
            model_target.get("maxSequenceLength", 256),
            1,
            256,
        )

    def _ip_adapter_scale(self, request: ImageRequest) -> float:
        # How strongly the reference image conditions the result (0 = ignore,
        # 1 = maximal). ~0.6 keeps subject identity while letting the prompt steer.
        try:
            scale = float(request.advanced.get("ipAdapterScale", 0.6))
        except (TypeError, ValueError):
            return 0.6
        return max(0.0, min(1.0, scale))


class SdxlDiffusersAdapter:
    """Stability AI Stable Diffusion XL base 1.0 via diffusers.StableDiffusionXLPipeline.

    Mirrors KolorsDiffusersAdapter (Kolors is built on the same SDXL UNet):
    HF-cache check, device/dtype selection, progress + cancel callbacks,
    incremental asset writing, and worker events for pipeline load + inference.
    Runs in the MAIN worker venv (diffusers + transformers) — no sidecar.

    SDXL uses real classifier-free guidance, so it honors the negative prompt and
    a non-zero guidance_scale (~7.0); the pipeline assembles the SDXL
    added_cond_kwargs (pooled text embeds + add_time_ids) itself. Two CLIP text
    encoders (no T5/ChatGLM), so there is no max_sequence_length knob. The
    checkpoint ships EulerDiscreteScheduler, which we keep (sampler/scheduler
    selection is epic 1753).

    Supports text-to-image (StableDiffusionXLPipeline) and img2img editing
    (StableDiffusionXLImg2ImgPipeline) on the same checkpoint, switched by
    request.mode.
    """

    id = "sdxl_diffusers"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._img2img_pipe: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_model: str | None = None
        # Whether the resident text pipe has the IP-Adapter (+ image encoder) loaded.
        # A plain-T2I pipe and an IP-Adapter pipe are not interchangeable, so this is
        # part of the text-pipe cache key.
        self._text_ip_adapter: bool = False
        self._loaded_lora_states: dict[str, LoraPipelineState] = {}

    @staticmethod
    def _use_ip_adapter(request: ImageRequest) -> bool:
        # IP-Adapter reference conditioning runs on the text-to-image pipeline only
        # (reference + img2img edit together is a future enhancement).
        return request.mode != "edit_image" and bool(request.reference_asset_id)

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._loaded_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        """Free any resident pipeline so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._text_pipe is None and self._img2img_pipe is None and getattr(self, "_pose_pipe", None) is None:
            return False
        self._evict_pipelines(importlib.import_module("torch"))
        return True

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
        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")
        model_target = MODEL_TARGETS.get(request.model, {})
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not an SDXL Diffusers target.")

        use_img2img = request.mode == "edit_image"
        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job["id"],
            adapter=self.id,
            loraCount=len(request.loras),
        )
        self._apply_loras(pipe, request)
        emit_worker_event("image_lora_apply_complete", jobId=job["id"], adapter=self.id)
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        label = f"{model_target['label']} edit" if use_img2img else model_target["label"]

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
                gpuMemory=gpu_memory_snapshot(torch, device),
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
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
                "repo": self._repo_for_request(request, model_target),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request, model_target),
                "realModelInference": True,
            },
        )

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
        diffusers = importlib.import_module("diffusers")
        repo = self._repo_for_request(request, model_target)
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        use_img2img = request.mode == "edit_image"
        use_ip_adapter = self._use_ip_adapter(request)
        ip_adapter = model_target.get("ipAdapter") or {}
        if use_ip_adapter and not ip_adapter:
            raise RuntimeError(
                f"{request.model} does not support reference-image (IP-Adapter) generation."
            )
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        cached_pipe = self._img2img_pipe if use_img2img else self._text_pipe
        # The text pipe's IP-Adapter state is part of its cache key: a plain-T2I pipe
        # cannot serve a reference job, and vice versa.
        text_state_matches = use_img2img or self._text_ip_adapter == use_ip_adapter
        if cached_pipe is not None and self._loaded_repo == repo and text_state_matches:
            self._loaded_model = request.model
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                model=request.model,
                repo=repo,
                device=device,
                componentDevices=pipeline_component_devices(cached_pipe),
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return cached_pipe

        # Hold only one pipeline per repo: evict stale pipes (other mode, stale repo,
        # or a text pipe whose IP-Adapter state no longer matches) before loading so
        # we don't keep two ~7GB SDXL pipelines resident.
        if self._loaded_repo and self._loaded_repo != repo:
            self._evict_pipelines(torch)
        elif use_img2img:
            if self._text_pipe is not None:
                self._text_pipe = None
                self._text_ip_adapter = False
                self._forget_loaded_loras("text")
                self._empty_cuda_cache(torch)
        else:
            if self._text_pipe is not None:
                self._text_pipe = None
                self._forget_loaded_loras("text")
                self._empty_cuda_cache(torch)
            if self._img2img_pipe is not None:
                self._img2img_pipe = None
                self._forget_loaded_loras("img2img")
                self._empty_cuda_cache(torch)

        pipeline_name = (
            "StableDiffusionXLImg2ImgPipeline" if use_img2img else "StableDiffusionXLPipeline"
        )
        pipeline_class = getattr(diffusers, pipeline_name, None)
        if pipeline_class is None:
            raise RuntimeError(
                f"The installed diffusers package does not expose {pipeline_name}. "
                "Install diffusers >= 0.19 for Stable Diffusion XL support."
            )

        cache_action = "Loading cached" if huggingface_repo_cache_exists(repo) else "Downloading"
        progress("loading_model", "loading_model", 0.2, f"{cache_action} {model_target['label']} model files.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useImg2img=use_img2img,
            useIpAdapter=use_ip_adapter,
            cpuOffload=cpu_offload,
            cached=cache_action == "Loading cached",
        )
        # SDXL base ships an fp16 variant alongside fp32; request it so
        # from_pretrained loads the half-precision weights.
        from_pretrained_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "variant": model_target.get("variant"),
        }
        if use_ip_adapter:
            # IP-Adapter (plus-face) needs the matching ViT-H CLIP image encoder,
            # shipped in the same repo at models/image_encoder. Loading it here +
            # passing image_encoder_folder=None to load_ip_adapter avoids a second
            # fetch attempt against the SDXL repo (which has no image_encoder).
            transformers = importlib.import_module("transformers")
            encoder_class = getattr(transformers, "CLIPVisionModelWithProjection", None)
            if encoder_class is None:
                raise RuntimeError(
                    "transformers does not expose CLIPVisionModelWithProjection, "
                    "required for the SDXL IP-Adapter image encoder."
                )
            from_pretrained_kwargs["image_encoder"] = encoder_class.from_pretrained(
                ip_adapter["repo"],
                subfolder=ip_adapter.get("encoderSubfolder", "models/image_encoder"),
                revision=ip_adapter.get("revision"),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        pipe = pipeline_class.from_pretrained(repo, **from_pretrained_kwargs)
        if use_ip_adapter:
            # image_encoder_folder=None: we already supplied the encoder above.
            pipe.load_ip_adapter(
                ip_adapter["repo"],
                subfolder=ip_adapter.get("subfolder", "sdxl_models"),
                weight_name=ip_adapter["weight"],
                revision=ip_adapter.get("revision"),
                image_encoder_folder=None,
            )
            pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        # VAE tiling keeps high-resolution decodes within memory.
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        elif hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        component_devices = verify_pipeline_on_device(
            pipe,
            requested_device=device,
            model_label=model_target["label"],
            allow_offload=offload_enabled,
        )
        emit_worker_event(
            "image_pipeline_on_device",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            requestedDevice=device,
            cpuOffload=offload_enabled,
            componentDevices=component_devices,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        if use_img2img:
            self._img2img_pipe = pipe
        else:
            self._text_pipe = pipe
            self._text_ip_adapter = use_ip_adapter
        self._loaded_repo = repo
        self._loaded_model = request.model
        return pipe

    def _evict_pipelines(self, torch: Any) -> None:
        self._text_pipe = None
        self._img2img_pipe = None
        self._text_ip_adapter = False
        self._loaded_repo = None
        self._loaded_model = None
        self._loaded_lora_states.clear()
        self._empty_cuda_cache(torch)

    def _empty_cuda_cache(self, torch: Any) -> None:
        release_inference_memory(torch)

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
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        model_target = MODEL_TARGETS[request.model]
        kwargs = {
            "prompt": request.prompt,
            # SDXL uses real classifier-free guidance, so the negative prompt is
            # honored (unlike the guidance-distilled FLUX path).
            "negative_prompt": request.negative_prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": self._num_inference_steps(request, model_target),
            "guidance_scale": self._guidance_scale(request, model_target),
            "generator": generator,
        }
        if request.mode == "edit_image":
            # StableDiffusionXLImg2ImgPipeline blends a source image; strength
            # controls how far the result moves from it (0 = unchanged, 1 = full
            # re-generation). It sizes the output from the source, so height/width
            # are dropped by filter_call_kwargs.
            kwargs["image"] = load_source_image(project_path, request)
            kwargs["strength"] = float(request.advanced.get("strength", 0.6))
        elif self._use_ip_adapter(request):
            # IP-Adapter conditions T2I on a reference image (style/identity). Vary
            # prompt/seed across the batch to get many images of the same subject.
            kwargs["ip_adapter_image"] = load_reference_image(project_path, request.reference_asset_id)
            if hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(self._ip_adapter_scale(request))
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**filter_call_kwargs(pipe, kwargs))
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        return output.images[0].convert("RGB")

    def _apply_loras(self, pipe: Any, request: ImageRequest) -> None:
        key = "img2img" if request.mode == "edit_image" else "text"
        model_target = MODEL_TARGETS.get(request.model, {})
        self._loaded_lora_states[key] = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            previous_state=self._loaded_lora_states.get(key),
        )

    def _forget_loaded_loras(self, key: str) -> None:
        self._loaded_lora_states.pop(key, None)

    def _repo_for_request(self, request: ImageRequest, model_target: dict[str, Any]) -> str:
        return request.advanced.get("modelRepo") or model_target["repo"]

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = model_target.get("guidanceScale", 7.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _ip_adapter_scale(self, request: ImageRequest) -> float:
        # How strongly the reference conditions the result (0 = ignore,
        # 1 = maximal). 0.7 holds plus-face identity while letting the prompt
        # steer scene/pose; identity-faithful likeness belongs to InstantID.
        try:
            scale = float(request.advanced.get("ipAdapterScale", 0.7))
        except (TypeError, ValueError):
            return 0.7
        return max(0.0, min(1.0, scale))


class ChromaDiffusersAdapter:
    """Lodestones Chroma1-HD / Base / Flash text-to-image via diffusers.ChromaPipeline.

    Near-clone of FluxDiffusersAdapter (Chroma1 is FLUX.1-schnell-derived and runs
    in the MAIN worker venv — confirmed by spike sc-1829), with two deltas:

    * Text encoder is T5-XXL **only** (no CLIP / no ``prompt_2``), so the single
      ``prompt`` field is all the pipeline takes.
    * Chroma exposes **real classifier-free guidance**: HD/Base honor a
      ``negative_prompt`` at guidance > 1; Flash bakes CFG (guidance 1.0) and
      ignores the negative prompt. We thread ``request.negative_prompt`` through;
      ``filter_call_kwargs`` drops it for any pipeline build that does not accept it.

    Text-to-image only (ChromaPipeline ships no img2img/inpaint variant).
    """

    id = "chroma_diffusers"

    def __init__(self) -> None:
        self._text_pipe: Any | None = None
        self._text_repo: str | None = None
        self._loaded_model: str | None = None
        self._loaded_lora_states: dict[str, LoraPipelineState] = {}

    def loaded_models(self) -> list[str]:
        return sorted({value for value in (self._text_repo, self._loaded_model) if value})

    def unload(self) -> bool:
        """Free the resident pipeline so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._text_pipe is None:
            return False
        self._text_pipe = None
        self._text_repo = None
        self._loaded_model = None
        self._loaded_lora_states.clear()
        self._empty_cuda_cache(importlib.import_module("torch"))
        return True

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
            raise RuntimeError(f"{request.model} is not a Chroma1 Diffusers target.")
        if request.mode == "edit_image":
            raise RuntimeError(f"{request.model} does not support image editing.")

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        pipe = self._load_pipeline(settings, request, model_target, progress=progress, job_id=job["id"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job["id"],
            adapter=self.id,
            loraCount=len(request.loras),
        )
        self._apply_loras(pipe, request)
        emit_worker_event("image_lora_apply_complete", jobId=job["id"], adapter=self.id)
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
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            try:
                image = self._run_pipeline(settings, pipe, request, seed, cancel_requested=cancel_requested)
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
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
                "repo": self._repo_for_request(request, model_target),
                "numInferenceSteps": self._num_inference_steps(request, model_target),
                "guidanceScale": self._guidance_scale(request, model_target),
                "maxSequenceLength": self._max_sequence_length(request, model_target),
                "negativePrompt": request.negative_prompt,
                "realModelInference": True,
            },
        )

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
        diffusers = importlib.import_module("diffusers")
        repo = self._repo_for_request(request, model_target)
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        cpu_offload = bool(request.advanced.get("cpuOffload", False))
        if self._text_pipe is not None and self._text_repo == repo:
            self._loaded_model = request.model
            progress("loading_model", "loading_model", 0.22, f"Using cached {model_target['label']}.")
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                model=request.model,
                repo=repo,
                device=device,
                componentDevices=pipeline_component_devices(self._text_pipe),
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return self._text_pipe
        if self._text_pipe is not None:
            self._text_pipe = None
            self._text_repo = None
            self._empty_cuda_cache(torch)
            self._forget_loaded_loras("text")

        pipeline_class = getattr(diffusers, "ChromaPipeline", None)
        if pipeline_class is None:
            raise RuntimeError(
                "The installed diffusers package does not expose ChromaPipeline. "
                "Install/upgrade diffusers (the worker pins diffusers git main) for Chroma1 support."
            )

        progress("loading_model", "loading_model", 0.2, f"Loading {model_target['label']} model files.")
        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            device=device,
            dtype=str(dtype),
            useImg2img=False,
            cpuOffload=cpu_offload,
            cached=huggingface_repo_cache_exists(repo),
        )
        pipe = pipeline_class.from_pretrained(repo, torch_dtype=dtype)
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            repo=repo,
            componentDevices=pipeline_component_devices(pipe),
        )
        offload_enabled = cpu_offload and hasattr(pipe, "enable_model_cpu_offload")
        if offload_enabled:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        # VAE tiling keeps high-resolution decodes within memory. Prefer the
        # current diffusers API (pipe.vae.enable_tiling) and fall back to the
        # deprecated pipeline-level shim for older builds.
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        elif hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        component_devices = verify_pipeline_on_device(
            pipe,
            requested_device=device,
            model_label=model_target["label"],
            allow_offload=offload_enabled,
        )
        emit_worker_event(
            "image_pipeline_on_device",
            jobId=job_id,
            adapter=self.id,
            model=request.model,
            requestedDevice=device,
            cpuOffload=offload_enabled,
            componentDevices=component_devices,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        self._text_pipe = pipe
        self._text_repo = repo
        self._loaded_model = request.model
        return pipe

    def _empty_cuda_cache(self, torch: Any) -> None:
        release_inference_memory(torch)

    def _run_pipeline(
        self,
        settings: WorkerSettings,
        pipe: Any,
        request: ImageRequest,
        seed: int,
        cancel_requested: CancelCallback | None = None,
    ) -> Image.Image:
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        generator = torch.Generator(device if device.startswith("cuda") else "cpu").manual_seed(seed)
        model_target = MODEL_TARGETS[request.model]
        kwargs = {
            "prompt": request.prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": self._num_inference_steps(request, model_target),
            # Chroma uses real CFG: HD/Base follow guidance (~3.0) and honor a
            # negative prompt; Flash bakes CFG (guidance 1.0) so the negative
            # prompt is a no-op. filter_call_kwargs drops negative_prompt for any
            # pipeline build that does not accept it.
            "guidance_scale": self._guidance_scale(request, model_target),
            "max_sequence_length": self._max_sequence_length(request, model_target),
            "generator": generator,
        }
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        step_callback = cancel_step_callback(pipe, cancel_requested)
        if step_callback is not None:
            kwargs["callback_on_step_end"] = step_callback
        output = pipe(**filter_call_kwargs(pipe, kwargs))
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Image generation canceled by user.")
        return output.images[0].convert("RGB")

    def _apply_loras(self, pipe: Any, request: ImageRequest) -> None:
        model_target = MODEL_TARGETS.get(request.model, {})
        self._loaded_lora_states["text"] = apply_loras_to_pipeline(
            pipe,
            request.loras,
            adapter_id=self.id,
            model_family=model_target.get("family"),
            previous_state=self._loaded_lora_states.get("text"),
        )

    def _forget_loaded_loras(self, key: str) -> None:
        self._loaded_lora_states.pop(key, None)

    def _repo_for_request(self, request: ImageRequest, model_target: dict[str, Any]) -> str:
        return request.advanced.get("modelRepo") or model_target["repo"]

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = model_target.get("guidanceScale", 3.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)

    def _max_sequence_length(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(
            request.advanced.get("maxSequenceLength"),
            model_target.get("maxSequenceLength", 512),
            1,
            512,
        )


# Lens trains on two base resolutions crossed with nine aspect ratios and expects
# `base_resolution` + `aspect_ratio` rather than free width/height. These mirror
# scene_worker/_vendor/lens/resolution.py so we can snap a SceneWorks W×H request
# onto the nearest trained bucket without importing the (diffusers-injecting) lens
# package just to read the table.
_LENS_BASE_RESOLUTIONS = (1024, 1440)
_LENS_ASPECT_RATIOS = (
    ("1:2", 1 / 2),
    ("9:16", 9 / 16),
    ("2:3", 2 / 3),
    ("3:4", 3 / 4),
    ("1:1", 1.0),
    ("4:3", 4 / 3),
    ("3:2", 3 / 2),
    ("16:9", 16 / 9),
    ("2:1", 2.0),
)
# (aspect_ratio, label) buckets for snap_to_aspect_bucket; preserves table order.
_LENS_ASPECT_BUCKETS = [(ratio, label) for label, ratio in _LENS_ASPECT_RATIOS]


_BucketT = TypeVar("_BucketT")


def snap_to_aspect_bucket(
    width: int, height: int, buckets: Iterable[tuple[float, _BucketT]]
) -> _BucketT:
    """Return the value of the bucket whose aspect ratio is closest to width/height
    in log-space. ``buckets`` is an iterable of (aspect_ratio, value) pairs; ties
    resolve to the first matching bucket, so callers pass tables in priority order.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    target = math.log(width / height)
    return min(buckets, key=lambda bucket: abs(target - math.log(bucket[0])))[1]


def lens_resolution_for(width: int, height: int) -> tuple[int, str]:
    """Snap a requested W×H to the nearest Lens (base_resolution, aspect_ratio).

    The base is chosen by total area against the geometric midpoint of the two
    bases' square areas (1024² and 1440²); the aspect ratio by closest log-ratio
    so portrait/landscape requests land on the matching bucket.
    """
    base = 1440 if max(1, int(width)) * max(1, int(height)) >= 1024 * 1440 else 1024
    return base, snap_to_aspect_bucket(width, height, _LENS_ASPECT_BUCKETS)


class LensTurboAdapter:
    """Microsoft Lens / Lens-Turbo text-to-image, run OUT-OF-PROCESS.

    Lens needs transformers 5.x (gpt-oss text encoder) + diffusers 0.38, which are
    incompatible with the main worker venv's transformers 4.x stack that native
    LTX-2.3 (ltx-core's Gemma-3 integration) requires. So Lens runs in a dedicated
    sidecar venv (``/opt/lens-venv``) via ``scene_worker/lens_runner.py``; this
    adapter only orchestrates that subprocess and writes the resulting PNGs through
    the shared asset writer. The vendored ``lens`` package (scene_worker/_vendor)
    is imported by the runner, not here.

    Text-to-image only (no edit/img2img). LoRAs (the `lens` family, trained by
    the `lens_lora` kernel) are resolved here and applied to the transformer in
    the sidecar via PeftAdapterMixin (sc-1587).
    """

    id = "lens_turbo"

    def __init__(self) -> None:
        # Sidecar scratch dir for the in-flight job, reaped by discard_temp_outputs
        # on force-cancel (os._exit skips the finally in generate). One job runs at
        # a time (sc-1719).
        self._scratch_dir: Path | None = None

    def discard_temp_outputs(self, job_id: str | None = None) -> None:
        """Reap the in-flight sidecar scratch dir only — filesystem-only.

        Called from generate's finally and from the force-cancel monitor thread
        right before os._exit, so it must stay filesystem-only (no torch/GPU; the
        main thread may be wedged in a native call)."""
        work_dir = self._scratch_dir
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._scratch_dir = None

    def loaded_models(self) -> list[str]:
        # The sidecar process loads and frees the model per job; nothing stays
        # resident in this (main-venv) process.
        return []

    @staticmethod
    def _lens_python() -> str:
        return os.getenv("SCENEWORKS_LENS_PYTHON", "/opt/lens-venv/bin/python")

    @staticmethod
    def _runner_path() -> Path:
        return Path(__file__).resolve().parent / "lens_runner.py"

    def _sidecar_available(self) -> bool:
        return Path(self._lens_python()).exists() and self._runner_path().exists()

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
        if request.mode == "edit_image":
            raise RuntimeError(f"{request.model} does not support image editing.")
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["lens_turbo"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a Lens target.")
        # Lens LoRAs (sc-1587) are trained on the base and applied to the
        # transformer inside the sidecar. Resolve + validate them in the main venv
        # so a bad path or incompatible family fails before we spawn the
        # subprocess; the sidecar only sees concrete file paths + weights.
        validate_lora_compatibility(
            request.loras, model_family=model_target.get("family"), adapter_id=self.id
        )
        lora_specs = normalize_lora_specs(request.loras)
        if not self._sidecar_available():
            raise RuntimeError(
                "Lens generation requires the isolated Lens sidecar venv. Rebuild the worker image with "
                "INCLUDE_LENS=1 (the Docker Compose default), or set SCENEWORKS_LENS_PYTHON to a Python "
                f"interpreter that has the lens stack installed (looked for {self._lens_python()})."
            )

        total = request.count
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        steps = self._num_inference_steps(request, model_target)
        guidance_scale = self._guidance_scale(request, model_target)
        base_resolution, aspect_ratio = lens_resolution_for(request.width, request.height)
        seeds = [resolve_seed(request.seed, request.prompt, index, request.seeds) for index in range(total)]
        lens_sampler, lens_scheduler, lens_shift = sampler_selection_from_advanced(request.advanced)
        torch = importlib.import_module("torch")
        device = select_torch_device(torch, getattr(settings, "gpu_id", None))
        # mxfp4 keeps the gpt-oss-20b text encoder small but needs CUDA + Triton
        # kernels, which exist only on NVIDIA. On MPS/CPU the encoder must load
        # dequantized to bf16 (transformers auto-falls back, but force it here so
        # a non-CUDA host never reaches the Triton path).
        disable_mxfp4 = bool(request.advanced.get("disableMxfp4", False)) or not device.startswith("cuda")

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']} (sidecar venv).")
        work_dir = Path(tempfile.mkdtemp(prefix="lens_sidecar_"))
        self._scratch_dir = work_dir
        try:
            images = self._run_sidecar(
                job_id=job["id"],
                work_dir=work_dir,
                label=model_target["label"],
                total=total,
                spec={
                    "repo": repo,
                    "prompt": request.prompt,
                    "negativePrompt": request.negative_prompt,
                    "baseResolution": base_resolution,
                    "aspectRatio": aspect_ratio,
                    "numInferenceSteps": steps,
                    "guidanceScale": guidance_scale,
                    "seeds": seeds,
                    "disableMxfp4": disable_mxfp4,
                    "cpuOffload": bool(request.advanced.get("cpuOffload", False)),
                    "dtype": request.advanced.get("dtype"),
                    "device": device,
                    "loras": [
                        {"path": lora.path, "weight": lora.weight, "name": lora.adapter_name}
                        for lora in lora_specs
                    ],
                    # Configurable sampler / scheduler (epic 1753 sc-1764). The
                    # sidecar's lens_runner swaps pipe.scheduler via
                    # apply_sampler before generation; the vendored Lens loop
                    # branches between its empirical mu+linear-sigma path
                    # (default) and the scheduler-native path (non-default).
                    **({"sampler": lens_sampler} if lens_sampler != "default" else {}),
                    **({"scheduler": lens_scheduler} if lens_scheduler != "default" else {}),
                    **({"schedulerShift": lens_shift} if lens_shift is not None else {}),
                },
                progress=progress,
                cancel_requested=cancel_requested,
            )

            def image_at_index(index: int) -> Image.Image:
                progress(
                    "running",
                    "generating",
                    image_batch_progress(index, total),
                    format_batch_running_message("Lens-Turbo", index, total),
                )
                with Image.open(images[index]) as handle:
                    return handle.convert("RGB")

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
                    "repo": repo,
                    "numInferenceSteps": steps,
                    "guidanceScale": guidance_scale,
                    "baseResolution": base_resolution,
                    "aspectRatio": aspect_ratio,
                    "textEncoderMxfp4": not disable_mxfp4,
                    "sidecarVenv": self._lens_python(),
                    "realModelInference": True,
                },
                settings=settings,
                job_id=job["id"],
            )
        finally:
            # The writer has read every PNG into the project by now; drop the
            # sidecar's scratch dir regardless of success/failure (also clears the
            # force-cancel registry).
            self.discard_temp_outputs(job["id"])

    def _run_sidecar(
        self,
        *,
        job_id: str,
        work_dir: Path,
        label: str,
        total: int,
        spec: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> list[str]:
        spec = {**spec, "outDir": str(work_dir)}
        spec_path = work_dir / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        stdout_log = work_dir / "stdout.log"
        cmd = [self._lens_python(), str(self._runner_path()), str(spec_path)]
        emit_worker_event(
            "lens_sidecar_start",
            jobId=job_id,
            adapter=self.id,
            repo=spec["repo"],
            imageCount=total,
            device=spec["device"],
            mxfp4=not spec["disableMxfp4"],
            sidecar=self._lens_python(),
        )
        progress("running", "generating", image_batch_progress(0, total), f"Running {label} ({total} image(s)).")
        # stdout -> file (avoids any pipe-fill deadlock); stderr inherits to the
        # worker log for diagnostics. Poll so the job stays cancelable; the
        # heartbeat thread keeps it alive during the (minutes-long) run.
        with stdout_log.open("w", encoding="utf-8") as out:
            # stderr merged into stdout.log so a native crash (SIGABRT/SIGSEGV)
            # leaves a partial traceback we can surface; result.json stays the
            # authoritative success channel (_read_result prefers it).
            proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT)
            while True:
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if cancel_requested():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise InterruptedError("Image generation canceled by user.")
        result = self._read_result(work_dir, stdout_log)
        if proc.returncode != 0 or "error" in result:
            error = result.get("error") or f"Lens sidecar exited with code {proc.returncode}."
            emit_worker_event(
                "lens_sidecar_failed",
                jobId=job_id,
                adapter=self.id,
                error=error,
                returnCode=proc.returncode,
            )
            raise RuntimeError(f"Lens generation failed in the sidecar venv: {error}")
        images = [str(path) for path in result.get("images", [])]
        if len(images) != total:
            raise RuntimeError(f"Lens sidecar produced {len(images)} image(s); expected {total}.")
        emit_worker_event("lens_sidecar_complete", jobId=job_id, adapter=self.id, imageCount=len(images))
        return images

    @staticmethod
    def _read_result(work_dir: Path, stdout_log: Path) -> dict[str, Any]:
        result_path = work_dir / "result.json"
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
        return {"error": "Lens sidecar produced no parseable result."}

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("steps"), model_target["steps"], 1, 80)

    def _guidance_scale(self, request: ImageRequest, model_target: dict[str, Any]) -> float:
        # Lens-Turbo is distilled for guidance_scale ~1.0 (no CFG); the base Lens
        # model uses ~5.0. The per-model default comes from MODEL_TARGETS so each
        # variant gets the right CFG when the request does not override it.
        default = model_target.get("guidanceScale", 1.0)
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return float(default)


class ProceduralImageAdapter:
    id = "procedural_preview"

    def loaded_models(self) -> list[str]:
        return []

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
        reject_loras_if_unsupported(request.loras, self.id)
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["z_image_turbo"])
        if request.mode == "edit_image" and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        def image_at_index(index: int) -> Image.Image:
            seed = resolve_seed(request.seed, request.prompt, index, request.seeds)
            progress(
                "running",
                "generating",
                image_batch_progress(index, request.count),
                f"Generated preview image {index + 1} of {request.count}.",
            )
            return render_preview_image(request, model_target, seed, index)

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
                "targetSteps": model_target["steps"],
                "previewRenderer": True,
            },
            settings=settings,
            job_id=job["id"],
        )


_SENSENOVA_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}
_SENSENOVA_ASPECT_BUCKETS = [(w / h, (w, h)) for (w, h) in _SENSENOVA_RESOLUTIONS.values()]


def sensenova_resolution_for(width: int, height: int) -> tuple[int, int]:
    """Snap a requested W×H to the nearest SenseNova-U1 trained bucket (by aspect ratio).

    SenseNova-U1 only renders well at its trained resolutions; off-bucket sizes
    degrade (upstream warns). Pick the bucket whose aspect ratio is closest in
    log-space so portrait/landscape requests land on the matching orientation.
    """
    return snap_to_aspect_bucket(width, height, _SENSENOVA_ASPECT_BUCKETS)


# SenseNova-U1 interleaved generation was trained at smaller buckets than plain
# text-to-image (e.g. 1:1 = 1536² vs 2048², 16:9 = 2048×1152 vs 2720×1536), so it
# has its own bucket set. Mirrors upstream examples/interleave/inference.py.
_INTERLEAVE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (1536, 1536),
    "16:9": (2048, 1152),
    "9:16": (1152, 2048),
    "3:2": (1888, 1248),
    "2:3": (1248, 1888),
    "4:3": (1760, 1312),
    "3:4": (1312, 1760),
    "1:2": (1088, 2144),
    "2:1": (2144, 1088),
    "1:3": (864, 2592),
    "3:1": (2592, 864),
}
_INTERLEAVE_ASPECT_BUCKETS = [(w / h, (w, h)) for (w, h) in _INTERLEAVE_RESOLUTIONS.values()]


def interleave_resolution_for(width: int, height: int) -> tuple[int, int]:
    """Snap a requested W×H to the nearest SenseNova-U1 *interleave* bucket by
    aspect ratio (log-space). Off-bucket sizes degrade, as upstream warns."""
    return snap_to_aspect_bucket(width, height, _INTERLEAVE_ASPECT_BUCKETS)


# Interleave inference requires a system prompt describing the think/no-think
# protocol the model was trained with; without it the model won't interleave
# correctly. Verbatim from upstream examples/interleave/inference.py (238d6cf).
_INTERLEAVE_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. "
    "You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a <think></think> block "
    "and place all reasoning inside it. You MUST interleave text with generated images "
    "using tags like <image1>, <image2>. Images can ONLY be generated between <think> and "
    "</think>, and may be referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer without "
    "reasoning. Do not use tags like <image1>, <image2>; present any images naturally "
    "alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. The answer "
    "may include text, images, or both. Match the user's language in both reasoning and the "
    "final answer."
)


class SenseNovaU1Adapter:
    """SenseNova-U1 unified multimodal model — text-to-image, run IN-PROCESS.

    NEO-unify (Qwen3-based Mixture-of-Transformers; no separate VAE/encoder).
    Unlike Lens, its deps (torch 2.8 / transformers 4.57.x / accelerate) match
    the main worker venv, so it loads in-process via the vendored ``sensenova_u1``
    package (scene_worker/_vendor): importing it registers the ``neo_chat`` model
    type, so ``AutoModel.from_pretrained`` resolves it with no trust_remote_code.
    Attention uses torch SDPA (flash-attn optional), so it runs on CUDA and MPS.

    Supports text-to-image and instruction-based editing (it2i); VQA and
    interleaved generation are not wired yet. The ``sensenova_u1_8b_fast``
    variant merges the upstream 8-step distill LoRA at load (see ``distillLora``
    in MODEL_TARGETS); user-supplied LoRAs are still rejected.
    """

    id = "sensenova_u1"

    # Pixel normalization from the upstream T2I example (examples/t2i/inference.py).
    _NORM_MEAN = (0.5, 0.5, 0.5)
    _NORM_STD = (0.5, 0.5, 0.5)

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._repo: str | None = None
        self._loaded_model: str | None = None
        # Identity of the distill LoRA merged into the cached model (or None for
        # the base model). The merge mutates weights in place, so this must be
        # part of the cache key — otherwise a fast-variant model would be reused
        # for the base 50-step variant (and vice versa).
        self._distill_lora_key: str | None = None

    def loaded_models(self) -> list[str]:
        return [self._loaded_model] if self._loaded_model else []

    def unload(self) -> bool:
        """Free the resident model so another family can load (cross-adapter
        eviction). Returns True if it actually freed something."""
        if self._model is None:
            return False
        self._model = None
        self._tokenizer = None
        self._repo = None
        self._distill_lora_key = None
        self._loaded_model = None
        release_inference_memory(importlib.import_module("torch"))
        return True

    @staticmethod
    def _ensure_vendor_on_path() -> None:
        vendor = str(Path(__file__).resolve().parent / "_vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)

    def _num_inference_steps(self, request: ImageRequest, model_target: dict[str, Any]) -> int:
        return safe_int(request.advanced.get("numInferenceSteps"), int(model_target.get("steps", 50)), 1, 100)

    @staticmethod
    def _guidance_scale(request: ImageRequest, model_target: dict[str, Any]) -> float:
        default = float(model_target.get("guidanceScale", 4.0))
        try:
            return float(request.advanced.get("guidanceScale", default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _image_guidance_scale(request: ImageRequest, default: float = 1.0) -> float:
        # Image-conditioning guidance for editing (it2i); upstream default is 1.0.
        # The character_image reference path raises the default to 1.5 so the model
        # is pulled harder toward the reference subject (the sc-2015 spike found
        # face-identity holds best when the reference is more heavily weighted).
        try:
            return float(request.advanced.get("imageGuidanceScale", default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _use_reference(request: ImageRequest) -> bool:
        # Character Studio reference path (sc-2016): feed the reference asset
        # through the same it2i_generate machinery the edit_image mode uses, but
        # the reference (vs source_asset_id) drives subject identity while the
        # prompt drives the new scene/pose. Mutually exclusive with edit_image
        # — character_image is a subject-variation flow, edit_image is a
        # localized modification flow. Mirrors QwenImageAdapter._use_reference.
        #
        # Honest tradeoff (sc-2015 hardware spike): SenseNova-U1's it2i path
        # preserves the reference's wardrobe + accessories + tattoos + hair
        # color very faithfully across scene variations, but face geometry
        # drifts unless the framing stays close to the reference (ArcFace
        # cosine 0.10–0.75 across cafe / pier / studio prompts). It's the
        # right pick when outfit consistency matters more than face fidelity;
        # InstantID-SDXL and PuLID-FLUX are the face-locked options.
        return request.mode == "character_image" and bool(request.reference_asset_id)

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
        reject_loras_if_unsupported(request.loras, self.id)
        model_target = MODEL_TARGETS.get(request.model, MODEL_TARGETS["sensenova_u1_8b"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{request.model} is not a SenseNova-U1 target.")
        is_edit = request.mode == "edit_image"
        is_character_image = self._use_reference(request)
        if is_edit and not model_supports_edit(request.model):
            raise RuntimeError(f"{request.model} does not support image editing.")

        torch = importlib.import_module("torch")
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, request.advanced.get("dtype"))
        repo = request.advanced.get("modelRepo") or model_target["repo"]
        distill_lora = model_target.get("distillLora") if isinstance(model_target.get("distillLora"), dict) else None
        steps = self._num_inference_steps(request, model_target)
        guidance_scale = self._guidance_scale(request, model_target)
        # Native timestep shift is the only sampling knob SenseNova exposes.
        # Image Studio surfaces it via the generic "schedulerShift" advanced
        # field (epic 1753 sc-1765); accept either name for back-compat.
        timestep_shift_raw = request.advanced.get("schedulerShift", request.advanced.get("timestepShift", 3.0))
        try:
            timestep_shift = float(timestep_shift_raw) if timestep_shift_raw is not None else 3.0
        except (TypeError, ValueError):
            timestep_shift = 3.0
        if timestep_shift <= 0.0:
            timestep_shift = 3.0
        # Default image-conditioning guidance is 1.0 for edit (upstream default),
        # 1.5 for character_image (pull harder toward the reference subject).
        img_guidance_scale = self._image_guidance_scale(request, default=1.5 if is_character_image else 1.0)
        width, height = sensenova_resolution_for(request.width, request.height)
        if is_edit:
            source_image = load_source_image(project_path, request)
        elif is_character_image:
            # Reference is loaded at the requested W×H so the model can match the
            # output bucket directly (matches the edit path's resize behavior).
            source_image = load_reference_image(project_path, request.reference_asset_id).resize(
                (width, height)
            )
        else:
            source_image = None

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        model, tokenizer = self._load_model(torch, repo, device, dtype, distill_lora=distill_lora, job_id=job["id"])
        self._loaded_model = request.model
        label = model_target["label"]
        # sc-2003 multi-backbone angle set: loop the 11 canonical angles when
        # advanced.angleSet is set on a character_image request, augmenting the
        # user's prompt per-angle. SenseNova-U1 has no landmark or face-ID
        # conditioning — the angle comes entirely from the prompt augment.
        # Spike-validated: mean ArcFace 0.29 across angles (lowest of the 4
        # angle-capable backbones for pure face fidelity, but uniquely
        # preserves the reference's wardrobe + accessories — the "character
        # continuity" tier in the picker).
        angle_set = is_character_image and bool(request.advanced.get("angleSet"))
        angles = list(CHARACTER_ANGLE_SET_ORDER) if angle_set else []
        total = len(angles) if angle_set else request.count
        set_seed = resolve_seed(request.seed, request.prompt, 0, request.seeds)

        def image_at_index(index: int) -> Image.Image:
            seed = set_seed if angle_set else resolve_seed(request.seed, request.prompt, index, request.seeds)
            angle = angles[index] if angle_set else None
            effective_prompt = augment_prompt_for_angle(request.prompt, angle) if angle else request.prompt
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
                device=device,
                resolution=f"{width}x{height}",
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            try:
                if is_edit or is_character_image:
                    # Both modes route through it2i_generate; the difference is
                    # which asset the source_image points at (source vs reference).
                    image = self._run_edit_inference(
                        torch, model, tokenizer, effective_prompt, source_image,
                        width, height, steps, guidance_scale, img_guidance_scale, timestep_shift, seed,
                    )
                else:
                    image = self._run_inference(
                        torch, model, tokenizer, effective_prompt,
                        width, height, steps, guidance_scale, timestep_shift, seed,
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
            emit_worker_event(
                "image_inference_complete",
                jobId=job["id"],
                adapter=self.id,
                imageIndex=index,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
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
                "repo": repo,
                "numInferenceSteps": steps,
                "guidanceScale": guidance_scale,
                **(
                    {"imageGuidanceScale": img_guidance_scale} if (is_edit or is_character_image) else {}
                ),
                "timestepShift": timestep_shift,
                "resolution": f"{width}x{height}",
                "realModelInference": True,
                **({"angleSet": True} if angle_set else {}),
            },
            settings=settings,
            job_id=job["id"],
        )

    def _load_model(
        self,
        torch: Any,
        repo: str,
        device: str,
        dtype: Any,
        *,
        distill_lora: dict[str, Any] | None = None,
        job_id: str,
    ) -> tuple[Any, Any]:
        lora_key = f"{distill_lora['repo']}/{distill_lora['file']}" if distill_lora else None
        if self._model is not None and self._repo == repo and self._distill_lora_key == lora_key:
            emit_worker_event(
                "image_pipeline_cache_hit",
                jobId=job_id,
                adapter=self.id,
                repo=repo,
                device=device,
                distillLora=lora_key,
                gpuMemory=gpu_memory_snapshot(torch, device),
            )
            return self._model, self._tokenizer
        if self._model is not None:
            self._model = None
            self._tokenizer = None
            self._repo = None
            self._distill_lora_key = None
            # gc.collect() before empty_cache(): the 8B-MoT we just dropped is
            # kept alive by its nn.Module reference cycles until the cyclic
            # collector runs, so on MPS a bare empty_cache() leaves the old model
            # resident alongside the replacement (base↔fast switches stacked to
            # tens of GB before this fix).
            release_inference_memory(torch)
        self._ensure_vendor_on_path()
        import sensenova_u1  # noqa: F401 — import registers the neo_chat model type
        from sensenova_u1.utils import load_model_and_tokenizer

        emit_worker_event(
            "image_pipeline_load_start",
            jobId=job_id,
            adapter=self.id,
            repo=repo,
            device=device,
            dtype=str(dtype),
            distillLora=lora_key,
            cached=huggingface_repo_cache_exists(repo),
        )
        model, tokenizer = load_model_and_tokenizer(repo, dtype=dtype, device=device)
        if distill_lora:
            self._merge_distill_lora(model, distill_lora, job_id=job_id)
        self._model = model
        self._tokenizer = tokenizer
        self._repo = repo
        self._distill_lora_key = lora_key
        emit_worker_event(
            "image_pipeline_load_complete",
            jobId=job_id,
            adapter=self.id,
            repo=repo,
            device=device,
            distillLora=lora_key,
        )
        return model, tokenizer

    def _merge_distill_lora(self, model: Any, distill_lora: dict[str, Any], *, job_id: str) -> None:
        """Resolve and merge the distill LoRA into the loaded model (in place).

        The ~0.4GB LoRA lives in a separate HF repo from the base weights, so it
        is fetched on demand: the local cache is checked first, then the hub. The
        vendored merge folds the delta into the model weights, so it survives the
        model cache (keyed by ``self._distill_lora_key``) with no per-call cost.
        """
        from sensenova_u1.utils import load_and_merge_lora_weight_from_safetensors

        repo = str(distill_lora["repo"])
        file_name = str(distill_lora["file"])
        emit_worker_event(
            "image_lora_apply_start",
            jobId=job_id,
            adapter=self.id,
            loraRepo=repo,
            loraFile=file_name,
        )
        lora_path = self._resolve_distill_lora_path(repo, file_name)
        load_and_merge_lora_weight_from_safetensors(model, str(lora_path))
        emit_worker_event(
            "image_lora_apply_complete",
            jobId=job_id,
            adapter=self.id,
            loraRepo=repo,
            loraFile=file_name,
            loraPath=str(lora_path),
        )

    @staticmethod
    def _resolve_distill_lora_path(repo: str, file_name: str) -> str:
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(repo_id=repo, filename=file_name, local_files_only=True)
        except Exception:
            return hf_hub_download(repo_id=repo, filename=file_name)

    def _run_inference(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        timestep_shift: float,
        seed: int,
    ) -> Image.Image:
        with torch.inference_mode():
            tensor = model.t2i_generate(
                tokenizer,
                prompt,
                image_size=(width, height),
                cfg_scale=guidance_scale,
                cfg_norm="none",
                timestep_shift=timestep_shift,
                cfg_interval=(0.0, 1.0),
                num_steps=steps,
                batch_size=1,
                seed=seed,
                think_mode=False,
            )
        return self._to_pil(torch, tensor)[0]

    def _run_edit_inference(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        prompt: str,
        source_image: Image.Image,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        img_guidance_scale: float,
        timestep_shift: float,
        seed: int,
    ) -> Image.Image:
        # Instruction-based editing (it2i): the source image is the conditioning
        # input; `image_size` is the output bucket. Defaults mirror the upstream
        # editing example (cfg 4.0 text / 1.0 image, 50 steps, shift 3.0).
        with torch.inference_mode():
            tensor = model.it2i_generate(
                tokenizer,
                prompt,
                [source_image],
                image_size=(width, height),
                cfg_scale=guidance_scale,
                img_cfg_scale=img_guidance_scale,
                cfg_norm="none",
                timestep_shift=timestep_shift,
                cfg_interval=(0.0, 1.0),
                num_steps=steps,
                batch_size=1,
                seed=seed,
                think_mode=False,
            )
        return self._to_pil(torch, tensor)[0]

    def answer_question(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        """Visual question answering (VQA): a text answer about a source image.

        Reuses the cached base model (the understanding side) via the model's
        ``chat`` path. Output is text, not an image asset, so this does not go
        through ImageAssetWriter — the answer is returned in the job result.
        """
        payload = job["payload"]
        project_id = payload["projectId"]
        source_asset_id = payload.get("sourceAssetId")
        question = str(payload.get("question") or "").strip()
        if not question:
            raise RuntimeError("Visual question answering requires a question.")
        model_id = payload.get("model", "sensenova_u1_8b")
        model_target = MODEL_TARGETS.get(model_id, MODEL_TARGETS["sensenova_u1_8b"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{model_id} is not a SenseNova-U1 target.")
        advanced = payload.get("advanced", {}) if isinstance(payload.get("advanced"), dict) else {}
        # VQA latency ~ output tokens (one model pass each) + input vision tokens
        # (prefill). Both default low for responsiveness and are tunable per request.
        max_new_tokens = safe_int(payload.get("maxNewTokens"), 256, 16, 2048)
        # Downscale the understanding input — vision tokens (and prefill cost) scale
        # with pixel count (~pixels/1024 tokens), and there's little perceptible
        # difference for question answering between ~768px and ~1024px. Default ~768²
        # (~576 tokens vs ~1024 at 1024²); tunable up via payload.maxImagePixels when a
        # question needs fine detail or in-image text.
        max_image_pixels = safe_int(payload.get("maxImagePixels"), 768 * 768, 256 * 256, 2048 * 2048)

        torch = importlib.import_module("torch")
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, advanced.get("dtype"))
        repo = advanced.get("modelRepo") or model_target["repo"]

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        # VQA uses the base understanding path — never the distilled generation LoRA.
        model, tokenizer = self._load_model(torch, repo, device, dtype, distill_lora=None, job_id=job["id"])
        self._loaded_model = model_id

        source_path = self._resolve_source_path(settings, project_id, source_asset_id)
        try:
            image = Image.open(source_path).convert("RGB")
        except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise RuntimeError(f"Source image could not be loaded safely: {source_path}") from exc

        if cancel_requested():
            raise InterruptedError("Visual question answering canceled by user.")
        progress("running", "generating", 0.6, "Analyzing image.")
        emit_worker_event(
            "image_vqa_start",
            jobId=job["id"],
            adapter=self.id,
            model=model_id,
            sourceAssetId=source_asset_id,
            device=device,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        answer = self._run_vqa(torch, model, tokenizer, image, question, device, max_new_tokens, max_image_pixels)
        emit_worker_event(
            "image_vqa_complete",
            jobId=job["id"],
            adapter=self.id,
            answerChars=len(answer),
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        return {
            "answer": answer,
            "question": question,
            "sourceAssetId": source_asset_id,
            "model": model_id,
            "realModelInference": True,
        }

    def _resolve_source_path(
        self,
        settings: WorkerSettings,
        project_id: str,
        source_asset_id: str | None,
    ) -> str:
        # Resolve only through the project sidecar/DB (find_asset_media_path constrains
        # the result to the project root). There is deliberately no client-supplied path
        # escape hatch: an arbitrary sourceImagePath would let a job read any file the
        # worker can open and, for VQA, return its contents to the caller.
        if not source_asset_id:
            raise RuntimeError("Visual question answering requires a source image asset.")
        project_path = shared_find_project_path(settings.data_dir / "recent-projects.json", project_id)
        return str(find_asset_media_path(project_path, source_asset_id))

    def _run_vqa(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        image: Image.Image,
        question: str,
        device: str,
        max_new_tokens: int,
        max_image_pixels: int,
    ) -> str:
        self._ensure_vendor_on_path()
        from sensenova_u1.models.neo_unify.utils import load_image_native

        pixel_values, grid_hw = load_image_native(image, max_pixels=int(max_image_pixels))
        pixel_values = pixel_values.to(device, dtype=model.dtype)
        grid_hw = grid_hw.to(device)
        generation_config = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
        with torch.inference_mode():
            # think=False skips the model's chain-of-thought so the budget goes to
            # the answer (otherwise reasoning fills the output and can truncate it).
            response = model.chat(tokenizer, pixel_values, question, generation_config, grid_hw=grid_hw, think=False)
        return self._strip_reasoning(str(response))

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Drop any ``<think>…</think>`` reasoning so only the answer is returned.

        Defensive backstop for the no-think prime: removes complete think blocks
        and any dangling/unclosed one (e.g. reasoning truncated by max_new_tokens).
        """
        import re

        cleaned = re.sub(r"(?s)<think>.*?</think>", "", text)
        cleaned = re.sub(r"(?s)<think>.*$", "", cleaned)
        return cleaned.strip()

    def _to_pil(self, torch: Any, batch: Any) -> list[Image.Image]:
        import numpy as np

        mean = torch.tensor(self._NORM_MEAN, device=batch.device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._NORM_STD, device=batch.device, dtype=torch.float32).view(1, 3, 1, 1)
        arr = (batch.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        arr = (arr * 255.0).round().astype(np.uint8)
        return [Image.fromarray(a) for a in arr]

    @staticmethod
    def _advanced_float(advanced: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(advanced.get(key, default))
        except (TypeError, ValueError):
            return default

    def generate_interleaved(
        self,
        *,
        settings: WorkerSettings,
        job: dict[str, Any],
        request: ImageRequest,
        project_path: Path,
        progress: ProgressCallback,
        cancel_requested: CancelCallback,
    ) -> dict[str, Any]:
        """Interleaved text-image generation (sc-1576): one model pass yields
        ordered text + images, persisted as a ``document`` asset. Reuses the cached
        base understanding+generation model — never the distilled generation LoRA.
        """
        payload = job["payload"]
        project_id = payload["projectId"]
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("Interleaved generation requires a prompt.")
        model_id = payload.get("model", "sensenova_u1_8b")
        model_target = MODEL_TARGETS.get(model_id, MODEL_TARGETS["sensenova_u1_8b"])
        if model_target.get("adapter") != self.id:
            raise RuntimeError(f"{model_id} is not a SenseNova-U1 target.")
        advanced = payload.get("advanced", {}) if isinstance(payload.get("advanced"), dict) else {}
        max_images = safe_int(payload.get("maxImages"), 6, 1, 10)
        width, height = interleave_resolution_for(
            safe_int(payload.get("width"), 2048, 256, 4096),
            safe_int(payload.get("height"), 1152, 256, 4096),
        )
        source_asset_ids = [str(asset) for asset in (payload.get("sourceAssetIds") or []) if asset]
        # Upstream interleave defaults (examples/interleave/inference.py @238d6cf).
        steps = safe_int(advanced.get("numInferenceSteps"), 50, 1, 100)
        cfg_scale = self._advanced_float(advanced, "guidanceScale", 4.0)
        img_cfg_scale = self._advanced_float(advanced, "imageGuidanceScale", 1.0)
        timestep_shift = self._advanced_float(advanced, "timestepShift", 3.0)
        max_new_tokens = safe_int(advanced.get("maxNewTokens"), 2048, 64, 8192)
        # Non-Think by default: the document is the deliverable, so skip the model's
        # chain-of-thought (mirrors the VQA think=False choice — "present images
        # naturally alongside the text"). Tunable; confirm on a real MPS run.
        think_mode = bool(advanced.get("thinkMode", False))
        # The think/no-think system prompt is exposed in the UI (prefilled with the
        # default); a blank/absent value falls back to _INTERLEAVE_SYSTEM_MESSAGE.
        system_message = str(advanced.get("systemMessage") or "").strip() or _INTERLEAVE_SYSTEM_MESSAGE
        seed = resolve_seed(payload.get("seed"), prompt, 0, None)

        torch = importlib.import_module("torch")
        require_inference_backend_for_gpu_worker(torch, settings.gpu_id)
        device = select_torch_device(torch, settings.gpu_id)
        activate_torch_device(torch, device)
        dtype = select_torch_dtype(torch, device, advanced.get("dtype"))
        repo = advanced.get("modelRepo") or model_target["repo"]

        progress("loading_model", "loading_model", 0.18, f"Loading {model_target['label']}.")
        model, tokenizer = self._load_model(torch, repo, device, dtype, distill_lora=None, job_id=job["id"])
        self._loaded_model = model_id

        input_images = self._load_input_images(project_path, source_asset_ids)

        if cancel_requested():
            raise InterruptedError("Interleaved generation canceled by user.")
        progress("running", "generating", 0.45, "Composing interleaved document.")
        emit_worker_event(
            "image_interleave_start",
            jobId=job["id"],
            adapter=self.id,
            model=model_id,
            device=device,
            resolution=f"{width}x{height}",
            maxImages=max_images,
            inputImages=len(input_images),
            thinkMode=think_mode,
            gpuMemory=gpu_memory_snapshot(torch, device),
        )
        generated_text, images = self._run_interleave(
            torch, model, tokenizer, prompt, input_images,
            width, height, steps, cfg_scale, img_cfg_scale, timestep_shift,
            max_images, max_new_tokens, think_mode, system_message, seed,
        )
        emit_worker_event(
            "image_interleave_complete",
            jobId=job["id"],
            adapter=self.id,
            imageCount=len(images),
            textChars=len(generated_text),
            gpuMemory=gpu_memory_snapshot(torch, device),
        )

        return self._write_interleaved_document(
            project_path=project_path,
            request=request,
            job=job,
            project_id=project_id,
            model_id=model_id,
            prompt=prompt,
            seed=seed,
            generated_text=generated_text,
            images=images,
            cancel_requested=cancel_requested,
            progress=progress,
            raw_settings={
                **advanced,
                "repo": repo,
                "numInferenceSteps": steps,
                "guidanceScale": cfg_scale,
                "imageGuidanceScale": img_cfg_scale,
                "timestepShift": timestep_shift,
                "maxImages": max_images,
                "maxNewTokens": max_new_tokens,
                "thinkMode": think_mode,
                "resolution": f"{width}x{height}",
                "realModelInference": True,
            },
        )

    def _load_input_images(
        self,
        project_path: Path,
        source_asset_ids: list[str],
    ) -> list[Image.Image]:
        if not source_asset_ids:
            return []
        images: list[Image.Image] = []
        for asset_id in source_asset_ids:
            path = find_asset_media_path(project_path, asset_id)
            try:
                images.append(Image.open(path).convert("RGB"))
            except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                raise RuntimeError(f"Source image could not be loaded safely: {path}") from exc
        return images

    def _run_interleave(
        self,
        torch: Any,
        model: Any,
        tokenizer: Any,
        prompt: str,
        input_images: list[Image.Image],
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        img_cfg_scale: float,
        timestep_shift: float,
        max_images: int,
        max_new_tokens: int,
        think_mode: bool,
        system_message: str,
        seed: int,
    ) -> tuple[str, list[Image.Image]]:
        generation_config = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
        with torch.inference_mode():
            text, image_tensors = model.interleave_gen(
                tokenizer,
                prompt,
                images=list(input_images),
                generation_config=generation_config,
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                cfg_norm="none",
                max_images=int(max_images),
                enable_timestep_shift=True,
                timestep_shift=timestep_shift,
                image_size=(width, height),
                cfg_interval=(0.0, 1.0),
                num_steps=int(steps),
                system_message=system_message,
                think_mode=think_mode,
                seed=int(seed),
            )
        pil_images: list[Image.Image] = []
        for tensor in image_tensors:
            pil_images.extend(self._to_pil(torch, tensor))
        return str(text), pil_images

    @staticmethod
    def _build_interleaved_segments(
        generated_text: str,
        image_writes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Split the model output on its inline ``<image>`` markers and slot the
        generated image assets in order: text[0], image[0], text[1], image[1], ….
        Reads the worker-reported image facts (assetId + mediaPath); Rust builds
        the image sidecars from the same facts (story 1656)."""
        parts = (generated_text or "").split("<image>")
        segments: list[dict[str, Any]] = []
        for index, part in enumerate(parts):
            text = part.strip()
            if text:
                segments.append({"type": "text", "text": text})
            if index < len(image_writes):
                write = image_writes[index]
                segments.append({"type": "image", "assetId": write["assetId"], "path": write["mediaPath"]})
        return segments

    def _write_interleaved_document(
        self,
        *,
        project_path: Path,
        request: ImageRequest,
        job: dict[str, Any],
        project_id: str,
        model_id: str,
        prompt: str,
        seed: int,
        generated_text: str,
        images: list[Image.Image],
        cancel_requested: CancelCallback,
        progress: ProgressCallback,
        raw_settings: dict[str, Any],
    ) -> dict[str, Any]:
        (project_path / "assets" / "documents").mkdir(parents=True, exist_ok=True)

        # Generated images persist as ordinary image assets — the worker saves the
        # PNG bytes + reports facts, and the Rust API builds + indexes their
        # sidecars (story 1656). The document references them in order.
        image_result = ImageAssetWriter().write_outputs(
            request=request,
            project_path=project_path,
            images=images,
            adapter_id=self.id,
            progress=lambda *_args, **_kwargs: None,
            cancel_requested=cancel_requested,
            raw_settings={**raw_settings, "interleaved": True},
            job_id=job["id"],
        )
        image_writes = image_result.get("assetWrites", [])
        generation_set_id = image_result.get("generationSetId")
        generation_set = image_result.get("generationSet")
        image_asset_ids = [write["assetId"] for write in image_writes]

        segments = self._build_interleaved_segments(generated_text, image_writes)

        created_at = utc_now()
        document_id = f"doc_{uuid4().hex}"
        media_rel = f"assets/documents/{document_id}.json"
        # The worker saves the document body (the "media"); the Rust API builds the
        # document sidecar + indexes project.db from the document fact below.
        write_json(
            project_path / media_rel,
            {
                "schemaVersion": 1,
                "id": document_id,
                "projectId": project_id,
                "jobId": job["id"],
                "model": model_id,
                "prompt": prompt,
                "createdAt": created_at,
                "segments": segments,
            },
        )

        asset_id = f"asset_{uuid4().hex}"
        document_write = {
            "type": "document",
            "assetId": asset_id,
            "mediaPath": media_rel,
            "mimeType": "application/json",
            "displayName": prompt[:56] or "Interleaved document",
            "createdAt": created_at,
            "mode": "interleave",
            "model": model_id,
            "adapter": self.id,
            "prompt": prompt,
            "negativePrompt": "",
            "seed": int(seed),
            "loras": [],
            "rawAdapterSettings": raw_settings,
            "maxImages": raw_settings.get("maxImages"),
            "resolution": raw_settings.get("resolution"),
            "imageCount": len(image_asset_ids),
            "parents": list(image_asset_ids),
        }
        asset_writes = [*image_writes, document_write]
        result = {
            "documentId": document_id,
            "documentAssetId": asset_id,
            "imageAssetIds": image_asset_ids,
            "segments": segments,
            "model": model_id,
            "realModelInference": True,
            "generationSetId": generation_set_id,
            "expectedCount": len(asset_writes),
            "generationSet": generation_set,
            "assetWrites": asset_writes,
        }
        progress("saving", "saving", 1.0, "Interleaved document saved.", result)
        return result


def create_image_adapter(
    job: dict[str, Any],
    adapters: dict[str, object] | None = None,
) -> (
    ProceduralImageAdapter
    | ZImageDiffusersAdapter
    | MlxZImageAdapter
    | QwenImageAdapter
    | MlxQwenAdapter
    | LensTurboAdapter
    | SenseNovaU1Adapter
    | FluxDiffusersAdapter
    | MlxFluxAdapter
    | MlxFlux2Adapter
    | KolorsDiffusersAdapter
    | SdxlDiffusersAdapter
    | MlxSdxlAdapter
    | ChromaDiffusersAdapter
    | "InstantIDAdapter"
    | "PuLIDFluxAdapter"
):
    payload = job.get("payload", {})
    requested = os.getenv("SCENEWORKS_IMAGE_ADAPTER", payload.get("adapter", "")).strip()
    if requested == "auto":
        requested = ""
    if requested in {"procedural", "procedural_preview"}:
        return adapters.get("procedural_preview") if adapters else ProceduralImageAdapter()
    # InstantID + PuLID-FLUX live in their own modules (each imports from this
    # one), so match by string id and lazily import on the no-adapters-dict path.
    if requested and requested not in {
        ZImageDiffusersAdapter.id,
        MlxZImageAdapter.id,
        QwenImageAdapter.id,
        MlxQwenAdapter.id,
        LensTurboAdapter.id,
        SenseNovaU1Adapter.id,
        FluxDiffusersAdapter.id,
        MlxFluxAdapter.id,
        MlxFlux2Adapter.id,
        KolorsDiffusersAdapter.id,
        SdxlDiffusersAdapter.id,
        MlxSdxlAdapter.id,
        ChromaDiffusersAdapter.id,
        "instantid_sdxl",
        "pulid_flux",
    }:
        raise RuntimeError(f"Unsupported SCENEWORKS_IMAGE_ADAPTER value: {requested}.")
    if requested == ZImageDiffusersAdapter.id:
        return adapters.get("z_image_diffusers") if adapters else ZImageDiffusersAdapter()
    if requested == MlxZImageAdapter.id:
        return adapters.get("mlx_z_image") if adapters else MlxZImageAdapter()
    if requested == QwenImageAdapter.id:
        return adapters.get("qwen_image") if adapters else QwenImageAdapter()
    if requested == MlxQwenAdapter.id:
        return adapters.get("mlx_qwen") if adapters else MlxQwenAdapter()
    if requested == LensTurboAdapter.id:
        return adapters.get("lens_turbo") if adapters else LensTurboAdapter()
    if requested == SenseNovaU1Adapter.id:
        return adapters.get("sensenova_u1") if adapters else SenseNovaU1Adapter()
    if requested == FluxDiffusersAdapter.id:
        return adapters.get("flux_diffusers") if adapters else FluxDiffusersAdapter()
    if requested == MlxFluxAdapter.id:
        return adapters.get("mlx_flux") if adapters else MlxFluxAdapter()
    if requested == MlxFlux2Adapter.id:
        return adapters.get("mlx_flux2") if adapters else MlxFlux2Adapter()
    if requested == KolorsDiffusersAdapter.id:
        return adapters.get("kolors_diffusers") if adapters else KolorsDiffusersAdapter()
    if requested == SdxlDiffusersAdapter.id:
        return adapters.get("sdxl_diffusers") if adapters else SdxlDiffusersAdapter()
    if requested == MlxSdxlAdapter.id:
        return adapters.get("mlx_sdxl") if adapters else MlxSdxlAdapter()
    if requested == ChromaDiffusersAdapter.id:
        return adapters.get("chroma_diffusers") if adapters else ChromaDiffusersAdapter()
    if requested == "instantid_sdxl":
        return _instantid_adapter(adapters)
    if requested == "pulid_flux":
        return _pulid_flux_adapter(adapters)
    model_target = MODEL_TARGETS.get(payload.get("model", "z_image_turbo"), {})
    if model_target.get("adapter") == ZImageDiffusersAdapter.id:
        # Z-Image auto-dispatch: prefer MlxZImageAdapter on macOS when its
        # sidecar venv (shared with FLUX / Qwen) is installed and the model is
        # supported (z_image_turbo only in v1). Otherwise fall back to the
        # torch path. SCENEWORKS_DISABLE_MLX_FLUX (shared opt-out) forces
        # torch. sc-2145.
        if _should_route_z_image_to_mlx(payload):
            return adapters.get("mlx_z_image") if adapters else MlxZImageAdapter()
        return adapters.get("z_image_diffusers") if adapters else ZImageDiffusersAdapter()
    if model_target.get("adapter") == QwenImageAdapter.id:
        # Qwen auto-dispatch: prefer MlxQwenAdapter on macOS when its sidecar venv
        # is installed AND the model is supported AND the job is plain T2I (no
        # reference asset, no edit_image — mflux QwenImageEdit needs spec
        # extension, deferred). Otherwise fall back to the torch path.
        # SCENEWORKS_DISABLE_MLX_FLUX (shared with the FLUX path) forces torch.
        # sc-1972.
        if _should_route_qwen_to_mlx(payload):
            return adapters.get("mlx_qwen") if adapters else MlxQwenAdapter()
        return adapters.get("qwen_image") if adapters else QwenImageAdapter()
    if model_target.get("adapter") == LensTurboAdapter.id:
        return adapters.get("lens_turbo") if adapters else LensTurboAdapter()
    if model_target.get("adapter") == SenseNovaU1Adapter.id:
        return adapters.get("sensenova_u1") if adapters else SenseNovaU1Adapter()
    if model_target.get("adapter") == FluxDiffusersAdapter.id:
        # FLUX auto-dispatch: prefer MlxFluxAdapter on macOS when its sidecar venv is
        # installed AND the model is supported AND the job has no reference image
        # (mflux has no FLUX IP-Adapter today). Otherwise fall back to the torch
        # path. SCENEWORKS_DISABLE_MLX_FLUX forces the torch path. sc-1970.
        if _should_route_flux_to_mlx(payload):
            return adapters.get("mlx_flux") if adapters else MlxFluxAdapter()
        return adapters.get("flux_diffusers") if adapters else FluxDiffusersAdapter()
    if model_target.get("adapter") == KolorsDiffusersAdapter.id:
        return adapters.get("kolors_diffusers") if adapters else KolorsDiffusersAdapter()
    if model_target.get("adapter") == MlxFlux2Adapter.id:
        # FLUX.2-klein has no torch fallback today — MlxFlux2Adapter is the
        # only adapter for the family. The frontend hides the model on
        # non-Mac hosts via the manifest `macOnly` flag; on any host the
        # adapter's own preflight (platform, sidecar venv, -kv requires
        # reference) raises a clear error if something is wrong, rather
        # than silently rendering the procedural placeholder. sc-2164.
        return adapters.get("mlx_flux2") if adapters else MlxFlux2Adapter()
    if model_target.get("adapter") == SdxlDiffusersAdapter.id:
        # SDXL auto-dispatch: prefer MlxSdxlAdapter on macOS when the vendored
        # mlx-examples is importable AND the model is supported AND the job
        # is plain T2I with no reference asset. Otherwise fall back to the
        # torch path. SCENEWORKS_DISABLE_MLX_SDXL forces torch even when MLX
        # is available (escape hatch for parity testing). sc-1975.
        if _should_route_sdxl_to_mlx(payload):
            return adapters.get("mlx_sdxl") if adapters else MlxSdxlAdapter()
        return adapters.get("sdxl_diffusers") if adapters else SdxlDiffusersAdapter()
    if model_target.get("adapter") == ChromaDiffusersAdapter.id:
        return adapters.get("chroma_diffusers") if adapters else ChromaDiffusersAdapter()
    if model_target.get("adapter") == "instantid_sdxl":
        return _instantid_adapter(adapters)
    if model_target.get("adapter") == "pulid_flux":
        return _pulid_flux_adapter(adapters)
    return adapters.get("procedural_preview") if adapters else ProceduralImageAdapter()


def _should_route_flux_to_mlx(payload: dict[str, Any]) -> bool:
    """Decide whether the FLUX auto-dispatch path should pick MlxFluxAdapter
    (mflux sidecar venv) over FluxDiffusersAdapter (torch/MPS). All checks must
    pass; any failure falls back to the torch path so we never regress it.
    sc-1970.

    Gates:
      1. SCENEWORKS_DISABLE_MLX_FLUX must be unset (escape hatch).
      2. Platform must be darwin (Apple Silicon — mflux/MLX is Mac-only).
      3. Model must be in MlxFluxAdapter._supported_models.
      4. Job must not request edit_image (mflux T2I-only for v1).
      5. Job must not have a reference asset (no FLUX IP-Adapter in mflux).
      6. The sidecar venv must exist (installed by provision_mlx_flux_venv on
         first launch). Spinning up a temp adapter just to read its predicate
         is cheap — no model load happens here.
    """
    if os.getenv("SCENEWORKS_DISABLE_MLX_FLUX", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if sys.platform != "darwin":
        return False
    model = payload.get("model")
    if model not in MlxFluxAdapter._supported_models:
        return False
    if payload.get("mode") == "edit_image":
        return False
    if payload.get("referenceAssetId"):
        return False
    return MlxFluxAdapter()._sidecar_available()


def _request_has_lokr_lora(payload: dict[str, Any]) -> bool:
    """True if any LoRA in the request is a LoKr/LyCORIS adapter. These apply only
    on the torch backends (the MLX merge math is LoRA-only), so the MLX routing
    gates fall back to torch when this is true (epic 2193) — the same graceful
    fallback a reference image triggers. Prefers the ``networkType`` recorded on
    the LoRA (zero I/O) and otherwise classifies the adapter's safetensors header
    (which also catches third-party LyCORIS files that carry no ``networkType``)."""

    for lora in payload.get("loras") or []:
        if not isinstance(lora, dict):
            continue
        recorded = lora.get("networkType") or (lora.get("compatibility") or {}).get("networkType")
        network_type = str(recorded or "").strip().lower()
        if network_type in ("lokr", "lycoris"):
            return True
        if not network_type:
            resolved = lora_path(lora)
            if resolved is not None and classify_adapter_network(resolved) in ("lokr", "lycoris"):
                return True
    return False


def _should_route_sdxl_to_mlx(payload: dict[str, Any]) -> bool:
    """Decide whether the SDXL auto-dispatch path should pick MlxSdxlAdapter
    (vendored mlx-examples in-proc) over SdxlDiffusersAdapter (torch). All
    checks must pass; any failure falls back to the torch path so we never
    regress it. sc-1975.

    NOTE: a *separate* env var (`SCENEWORKS_DISABLE_MLX_SDXL`) from the mflux
    family's `SCENEWORKS_DISABLE_MLX_FLUX`, because the MLX SDXL stack is
    structurally different (in-process vendored, not subprocess + sidecar
    venv). Each backend gets its own escape hatch.

    Gates:
      1. SCENEWORKS_DISABLE_MLX_SDXL unset.
      2. Platform == darwin (mlx + mlx-examples are Apple-only).
      3. Model in MlxSdxlAdapter._supported_models (just `sdxl` in v1).
      4. mode != "edit_image" (img2img isn't vendored).
      5. No referenceAssetId (no IP-Adapter in the MLX path).
      6. The vendored mlx_sd package + mlx itself must import (the
         requirements-mlx.txt install gate).
      7. No LoKr LoRA in the request — LoKr is torch-only (epic 2193); a LoKr
         job falls back to the torch path the same way a reference image does.
    """
    if os.getenv("SCENEWORKS_DISABLE_MLX_SDXL", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if _request_has_lokr_lora(payload):
        return False
    if sys.platform != "darwin":
        return False
    model = payload.get("model")
    if model not in MlxSdxlAdapter._supported_models:
        return False
    if payload.get("mode") == "edit_image":
        return False
    if payload.get("referenceAssetId"):
        return False
    return MlxSdxlAdapter._mlx_sd_available()


def _should_route_z_image_to_mlx(payload: dict[str, Any]) -> bool:
    """Decide whether the Z-Image auto-dispatch path should pick MlxZImageAdapter
    (shared mflux sidecar venv) over ZImageDiffusersAdapter (torch / diffusers).
    Mirrors `_should_route_flux_to_mlx` / `_should_route_qwen_to_mlx` exactly,
    except only `z_image_turbo` is wired in v1 (mflux also has `z_image` base /
    full but SceneWorks only catalogs Turbo today — sc-2005 cleanup). sc-2145.

    Gates:
      1. SCENEWORKS_DISABLE_MLX_FLUX unset (shared opt-out — one env var per
         sidecar venv, not per mflux family).
      2. Platform == darwin.
      3. Model in MlxZImageAdapter._supported_models.
      4. mode != "edit_image".
      5. No referenceAssetId (Z-Image has no reference path on either
         backend; the flag is checked symmetrically with the other mflux
         families).
      6. Sidecar venv exists.
    """
    if os.getenv("SCENEWORKS_DISABLE_MLX_FLUX", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if sys.platform != "darwin":
        return False
    model = payload.get("model")
    if model not in MlxZImageAdapter._supported_models:
        return False
    if payload.get("mode") == "edit_image":
        return False
    if payload.get("referenceAssetId"):
        return False
    # LoKr is torch-only (epic 2193); fall back to the torch path for a LoKr job.
    if _request_has_lokr_lora(payload):
        return False
    return MlxZImageAdapter()._sidecar_available()


def _should_route_flux2_to_mlx(payload: dict[str, Any]) -> bool:
    """Decide whether MlxFlux2Adapter can handle a FLUX.2-klein job.

    Unlike the other ``_should_route_*_to_mlx`` predicates, this is NOT an
    "MLX vs torch" choice — FLUX.2-klein has no torch backend in SceneWorks
    today, so MlxFlux2Adapter is the only adapter for the family. If this
    predicate returns False on a flux2-klein job, the dispatch hands it to
    the procedural preview placeholder and the frontend should have already
    hidden the model on the host (manifest macOnly flag).

    Gates:
      1. SCENEWORKS_DISABLE_MLX_FLUX unset (shared opt-out with the other
         mflux family — one env var per sidecar venv, not per family).
      2. Platform == darwin (mflux/MLX is Apple-only).
      3. Model in MlxFlux2Adapter._supported_models.
      4. Sidecar venv exists.
    Both ids route by reference presence inside the runner (txt2img vs edit);
    -kv no longer requires a reference — its txt2img path is on par with the
    base 9B (sc-2164, sc-2173).
    """
    if os.getenv("SCENEWORKS_DISABLE_MLX_FLUX", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if sys.platform != "darwin":
        return False
    model = payload.get("model")
    if model not in MlxFlux2Adapter._supported_models:
        return False
    return MlxFlux2Adapter()._sidecar_available()


def _should_route_qwen_to_mlx(payload: dict[str, Any]) -> bool:
    """Decide whether the Qwen-Image auto-dispatch path should pick MlxQwenAdapter
    (mflux sidecar venv) over QwenImageAdapter (torch/diffusers). All checks must
    pass; any failure falls back to the torch path so we never regress it.
    sc-1972.

    Mirrors `_should_route_flux_to_mlx` exactly except for the supported-models
    set: only `qwen_image` is wired today. `qwen_image_edit` and
    `qwen_image_edit_2509` need additional spec/runner work for mflux's
    QwenImageEdit.image_paths interface, so they stay on the torch path.

    Gates:
      1. SCENEWORKS_DISABLE_MLX_FLUX unset (shared opt-out — one env var per
         sidecar venv, not per mflux family).
      2. Platform == darwin.
      3. Model in MlxQwenAdapter._supported_models.
      4. mode != "edit_image".
      5. No referenceAssetId (character/reference flow is on the torch
         path while QwenImageEdit threading is deferred).
      6. Sidecar venv exists.
    """
    if os.getenv("SCENEWORKS_DISABLE_MLX_FLUX", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if sys.platform != "darwin":
        return False
    model = payload.get("model")
    if model not in MlxQwenAdapter._supported_models:
        return False
    if payload.get("mode") == "edit_image":
        return False
    if payload.get("referenceAssetId"):
        return False
    return MlxQwenAdapter()._sidecar_available()


def _instantid_adapter(adapters: dict[str, object] | None) -> "InstantIDAdapter":
    """Resolve the registered InstantID adapter, or lazily construct one when no
    runtime adapters dict is supplied (tests / direct calls). Kept separate to
    avoid a module-level import cycle (instantid_adapter imports from here)."""
    if adapters and "instantid_sdxl" in adapters:
        return adapters["instantid_sdxl"]
    from .instantid_adapter import InstantIDAdapter

    return InstantIDAdapter()


def _pulid_flux_adapter(adapters: dict[str, object] | None) -> "PuLIDFluxAdapter":
    """Resolve the registered PuLID-FLUX adapter, or lazily construct one when no
    runtime adapters dict is supplied (tests / direct calls). Kept separate for
    the same import-cycle reason as `_instantid_adapter` (pulid_flux_adapter
    imports from here)."""
    if adapters and "pulid_flux" in adapters:
        return adapters["pulid_flux"]
    from .pulid_flux_adapter import PuLIDFluxAdapter

    return PuLIDFluxAdapter()


def model_supports_edit(model_id: str) -> bool:
    return bool(MODEL_TARGETS.get(model_id, {}).get("supportsEdit"))


def resolve_seed(seed: int | None, prompt: str, index: int, seeds: list[int] | None = None) -> int:
    if seed is not None:
        return int(seed) + index
    if seeds and index < len(seeds):
        return int(seeds[index])
    digest = hashlib.sha256(f"{prompt}:{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def image_batch_progress(completed_count: int, total: int) -> float:
    safe_total = max(1, total)
    bounded_count = min(max(0, completed_count), safe_total)
    return 0.78 + (bounded_count / safe_total) * 0.17


def select_torch_device(torch: Any, gpu_id: str | None = None) -> str:
    # An explicit "cpu" forces CPU on any platform, including Apple Silicon where
    # MPS would otherwise be picked (honors SCENEWORKS_GPU_ID=cpu, sc-1335).
    if str(gpu_id or "").strip().lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        gpu_id = str(gpu_id or "").strip()
        if gpu_id.isdigit():
            try:
                device_count = int(torch.cuda.device_count())
            except (AttributeError, TypeError, ValueError):
                device_count = 0
            physical_index = int(gpu_id)
            if device_count > 1 and physical_index < device_count:
                return f"cuda:{physical_index}"
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Route the few diffusers ops that lack an MPS kernel to CPU instead of
        # erroring out. Set only when MPS is actually selected — never on CUDA
        # or CPU hosts (sc-1332).
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def empty_torch_cache(torch: Any) -> None:
    """Release cached accelerator memory on whichever backend is active.

    Clears the CUDA allocator cache on NVIDIA and the MPS allocator cache on
    Apple Silicon; a no-op on CPU-only hosts (sc-1332).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps and mps.is_available():
        mps_backend = getattr(torch, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "empty_cache"):
            mps_backend.empty_cache()


def release_inference_memory(torch: Any) -> None:
    """Collect dropped references, then return cached accelerator blocks to the OS.

    A bare ``empty_cache()`` reclaims nothing while a just-dropped model/pipeline
    is still kept alive by reference cycles in its ``nn.Module`` graph (the cyclic
    collector has not run yet), and the MPS/CUDA caching allocator only returns
    freed blocks to the OS on ``empty_cache()``. Both steps are required, in order:
    collect first, then empty — otherwise an evicted multi-GB model lingers
    alongside its replacement (the cross-model accumulation that pins MPS memory).
    """
    gc.collect()
    empty_torch_cache(torch)


def release_image_worker_memory() -> None:
    """Drop transient inference buffers after a job WITHOUT evicting cached models.

    Returns the post-generation activation pool (tens of GB at large resolutions)
    to the OS so an idle worker does not sit at peak memory, while cached
    pipelines/models stay resident for fast reuse. The just-finished job's tensors
    are already unreferenced here, so ``empty_cache()`` reclaims them; the model,
    still held by the adapter, is untouched. No-op when torch is unavailable.
    """
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return
    release_inference_memory(torch)


def evict_other_image_adapters(adapters: dict[str, object], keep_id: str) -> None:
    """Enforce a single resident image model: unload every adapter except keep_id.

    Each adapter is a long-lived singleton that only evicts its OWN previous model
    on a within-family switch, so without this, running e.g. Qwen then SenseNova
    leaves both multi-GB models resident. Called before a job loads its model so
    the previous family's weights are freed first (no transient double residency).
    Adapters with nothing resident (procedural, out-of-process Lens) expose no
    unload and are skipped.
    """
    freed: list[str] = []
    for adapter_id, adapter in adapters.items():
        if adapter_id == keep_id:
            continue
        unload = getattr(adapter, "unload", None)
        if not callable(unload):
            continue
        try:
            if unload():
                freed.append(adapter_id)
        except Exception as exc:  # noqa: BLE001 - never let cleanup abort a job
            emit_worker_event("image_adapter_unload_failed", adapter=adapter_id, error=str(exc))
    if freed:
        emit_worker_event("image_adapters_evicted", keep=keep_id, evicted=freed)


def torch_inference_backend_available(torch: Any | None = None) -> bool:
    """True when a CUDA or MPS inference backend is usable.

    Callers that already imported torch pass it in; callers that haven't (e.g.
    worker capability registration before torch setup) omit it, so we import torch
    defensively and treat any failure as "no backend".
    """
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except Exception:
            return False
    try:
        if bool(torch.cuda.is_available()):
            return True
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps and mps.is_available())
    except Exception:
        return False


def require_inference_backend_for_gpu_worker(torch: Any, gpu_id: str | None) -> None:
    requested = str(gpu_id or "").strip().lower()
    if requested != "cpu" and not torch_inference_backend_available(torch):
        raise RuntimeError(
            "CUDA-enabled PyTorch is not available in this GPU worker. "
            "Rebuild the worker with a CUDA PyTorch wheel, for example "
            "`docker compose build worker --no-cache`, then restart the worker."
        )


def activate_torch_device(torch: Any, device: str) -> None:
    if device.startswith("cuda:") and hasattr(torch.cuda, "set_device"):
        torch.cuda.set_device(device)


def select_torch_dtype(torch: Any, device: str, requested: Any) -> Any:
    if requested == "float16":
        return torch.float16
    if requested == "float32" or device == "cpu":
        return torch.float32
    # bfloat16 for both CUDA and MPS. On MPS, float16 overflows to NaN and yields
    # all-black images — the denoiser latents go NaN before the VAE, so upcasting
    # only the VAE does not help. bfloat16 keeps float32's exponent range (no
    # overflow) at half float32's memory and renders correctly + fast on Apple
    # Silicon. Verified on an M-series Mac with Z-Image-Turbo: fp16 = black,
    # bf16 = OK (~18s), fp32 = OK but ~2x memory and slower. Explicit
    # float16/float32 requests above still win (sc-1336).
    return torch.bfloat16


def load_source_image(project_path: Path, request: ImageRequest) -> Image.Image:
    # Resolve only through the project sidecar/DB (find_asset_media_path constrains the
    # result to the project root). No client-supplied path escape hatch: an arbitrary
    # sourceImagePath would let an edit job read any file the worker can open.
    if not request.source_asset_id:
        raise RuntimeError("Image edit jobs require a source image asset.")
    source_path = find_asset_media_path(project_path, request.source_asset_id)
    try:
        image = Image.open(source_path).convert("RGB")
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise RuntimeError(f"Source image could not be loaded safely: {source_path}") from exc
    return image.resize((request.width, request.height))


def load_reference_image(project_path: Path, reference_asset_id: str) -> Image.Image:
    # IP-Adapter reference image (style/identity conditioning). Resolved only through
    # the project sidecar/DB (find_asset_media_path constrains to the project root) —
    # no client-supplied path escape hatch. Returned at native resolution: the
    # IP-Adapter's CLIP image encoder does its own preprocessing, so resizing it to the
    # output W×H here would distort the conditioning.
    if not reference_asset_id:
        raise RuntimeError("Reference-image generation requires a reference image asset.")
    reference_path = find_asset_media_path(project_path, reference_asset_id)
    try:
        return Image.open(reference_path).convert("RGB")
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise RuntimeError(
            f"Reference image could not be loaded safely: {reference_path}"
        ) from exc


def find_asset_media_path(project_path: Path, asset_id: str) -> Path:
    sidecar_path = find_asset_sidecar_path(project_path, asset_id)
    if sidecar_path is not None:
        asset = read_json(sidecar_path)
        media_path = project_path / asset.get("file", {}).get("path", "")
        if media_path.exists():
            return media_path
        raise RuntimeError(f"Source image file is missing for asset {asset_id}.")
    raise RuntimeError(f"Source image asset not found: {asset_id}.")


def render_preview_image(request: ImageRequest, model_target: dict[str, Any], seed: int, index: int) -> Image.Image:
    import numpy as np

    width = min(request.width, 1280)
    height = min(request.height, 1280)
    digest = hashlib.sha256(f"{request.prompt}:{request.style_preset}:{seed}".encode("utf-8")).digest()
    base = np.array([digest[0], digest[1], digest[2]], dtype=np.float32)
    accent = np.array([digest[9], digest[10], digest[11]], dtype=np.float32)
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    mix = x * 0.56 + y * 0.44
    xi = np.arange(width, dtype=np.uint32)[None, :]
    yi = np.arange(height, dtype=np.uint32)[:, None]
    wave = ((xi * digest[3] + yi * digest[4] + seed) % 255).astype(np.float32) / 255
    pixels = base * (1 - mix[..., None]) + accent * mix[..., None] * 0.85 + wave[..., None] * 34
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, height * 0.68, width, height), fill=(12, 12, 12, 168))
    draw.rectangle((0, 0, width, 84), fill=(12, 12, 12, 118))
    font = ImageFont.load_default()
    title = f"{model_target['label']} preview #{index + 1}"
    draw.text((28, 26), title, fill=(250, 241, 220, 255), font=font)
    draw.text((28, 50), f"{request.mode.replace('_', ' ')} | seed {seed}", fill=(194, 235, 226, 255), font=font)

    text = request.prompt.strip() or "Untitled prompt"
    y = int(height * 0.7) + 24
    for line in wrap(text, width=max(28, width // 14))[:8]:
        draw.text((28, y), line, fill=(255, 255, 255, 242), font=font)
        y += 18
    return image
