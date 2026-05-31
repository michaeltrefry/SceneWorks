"""sc-2475 spike — SDXL/RealVisXL masked inpaint feasibility on MPS (GO/NO-GO for sc-2476).

Question: does ``StableDiffusionXLInpaintPipeline``, loaded from the SDXL / RealVisXL
single-file checkpoint SceneWorks already ships, produce good **masked inpaint on M5 Max
(MPS, bf16)** — the masked region follows the prompt, pixels OUTSIDE the mask are
preserved, seams blend — at acceptable peak memory + time? It informs the recipe
(strength / steps / guidance / mask feather) that sc-2476 will wire into the SDXL adapter
as a third pipeline branch beside the existing ``StableDiffusionXLImg2ImgPipeline``
(image_adapters.py ~:5527/:5669).

Why this beachhead: the edit-capable models today are all img2img or instruction-edit —
none take a binary mask (sc-2436 spike finding). A regular SDXL/RealVisXL checkpoint has a
4-channel UNet; diffusers' inpaint pipeline still runs it via the **legacy mask-blended
img2img path** (per-step latent blend that keeps outside-mask pixels), so NO new checkpoint
download is required for v1. Pass ``--inpaint-checkpoint`` to also A/B a dedicated 9-channel
SDXL-inpaint UNet (e.g. ``diffusers/stable-diffusion-xl-1.0-inpainting-0.1``) for quality.

Run with the SceneWorks desktop venv (has torch/diffusers):
  "/Users/michael/Library/Application Support/SceneWorks/python/venv/bin/python" \
      scripts/spikes/sc2475_sdxl_inpaint_spike.py \
      --checkpoint /path/to/RealVisXL_V4.0.safetensors \
      --source /path/to/photo.png --prompt "a red sports car"

Defaults pick a checkpoint from the HF cache / data models dir and synthesize a centered
elliptical mask, so it runs with just ``--checkpoint`` + ``--source`` (or fully argless if
the defaults resolve). Outputs PNGs + a JSON summary under --out (default /tmp/sc2475_inpaint).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SCENEWORKS_GPU_ID", "mps")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _peak_mem_gb(device: str) -> float | None:
    """Best-effort peak allocation in GB for the active device."""
    try:
        if device == "mps":
            return torch.mps.driver_allocated_memory() / 1e9
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        return None
    return None


def _load_source(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # Square-fit to `size` so the mask geometry is predictable for the probe.
    return img.resize((size, size), Image.LANCZOS)


def _centered_ellipse_mask(size: int, feather: int) -> Image.Image:
    """White (=inpaint) centered ellipse on black, optionally feathered. PIL 'L'."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    pad = size // 6
    draw.ellipse([pad, pad, size - pad, size - pad], fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def _load_pipeline(checkpoint: str, device: str):
    from diffusers import StableDiffusionXLInpaintPipeline

    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    ckpt_path = Path(checkpoint)
    if ckpt_path.exists() and ckpt_path.suffix in (".safetensors", ".ckpt"):
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(checkpoint, torch_dtype=dtype)
    else:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(checkpoint, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _preservation_stats(source: Image.Image, result: Image.Image, mask: Image.Image) -> dict:
    """Mean abs RGB diff (0..255) inside vs outside the mask. True inpaint keeps OUTSIDE ~0
    and changes INSIDE; a large outside-diff means the pipeline rewrote the whole frame."""
    src = np.asarray(source.convert("RGB"), dtype=np.float32)
    res = np.asarray(result.convert("RGB").resize(source.size), dtype=np.float32)
    m = np.asarray(mask.convert("L").resize(source.size), dtype=np.float32) / 255.0
    inside = m > 0.5
    outside = ~inside
    diff = np.abs(res - src).mean(axis=2)
    return {
        "insideMeanDiff": round(float(diff[inside].mean()) if inside.any() else 0.0, 2),
        "outsideMeanDiff": round(float(diff[outside].mean()) if outside.any() else 0.0, 2),
        "insidePixels": int(inside.sum()),
        "outsidePixels": int(outside.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SC2475_CHECKPOINT", ""),
        help="SDXL/RealVisXL single-file .safetensors path (4-ch UNet → legacy mask-blend) "
        "or an HF repo id. REQUIRED if no default resolves.",
    )
    parser.add_argument(
        "--inpaint-checkpoint",
        default="",
        help="Optional dedicated 9-ch SDXL-inpaint checkpoint to A/B for quality "
        "(e.g. diffusers/stable-diffusion-xl-1.0-inpainting-0.1).",
    )
    parser.add_argument("--source", default="", help="Source image (square-fit to --size).")
    parser.add_argument("--mask", default="", help="Optional mask PNG (white=edit). Else a centered ellipse.")
    parser.add_argument("--prompt", default="a vibrant bouquet of red roses, sharp focus")
    parser.add_argument("--negative-prompt", default="blurry, lowres, deformed")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--feather", type=int, default=12, help="Mask Gaussian blur radius (soft edge).")
    parser.add_argument("--strengths", default="0.75,1.0", help="Comma-separated denoise strengths to sweep.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="/tmp/sc2475_inpaint")
    args = parser.parse_args()

    if not args.checkpoint:
        print(
            "ERROR: pass --checkpoint <RealVisXL/SDXL .safetensors path or HF id>.\n"
            "  Tip: your installed SDXL/RealVisXL checkpoint usually lives under\n"
            "  ~/Library/Application Support/SceneWorks/data/models/ or the HF hub cache.",
            file=sys.stderr,
        )
        return 2
    if not args.source:
        print("ERROR: pass --source <image path>.", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = _device()
    size = args.size

    source = _load_source(Path(args.source), size)
    source.save(out / "source.png")
    if args.mask:
        mask = Image.open(args.mask).convert("L").resize((size, size))
    else:
        mask = _centered_ellipse_mask(size, args.feather)
    mask.save(out / "mask.png")

    strengths = [float(s) for s in args.strengths.split(",") if s.strip()]
    checkpoints = [("base", args.checkpoint)]
    if args.inpaint_checkpoint:
        checkpoints.append(("inpaint9ch", args.inpaint_checkpoint))

    summary: dict = {
        "device": device,
        "size": size,
        "steps": args.steps,
        "guidance": args.guidance,
        "feather": args.feather,
        "prompt": args.prompt,
        "runs": [],
    }

    for ckpt_label, ckpt in checkpoints:
        print(f"\n=== loading {ckpt_label}: {ckpt} on {device} ===", flush=True)
        t_load = time.time()
        try:
            pipe = _load_pipeline(ckpt, device)
        except Exception as exc:  # noqa: BLE001
            summary["runs"].append({"checkpoint": ckpt_label, "error": f"load failed: {exc}"})
            print(f"  load FAILED: {exc}", file=sys.stderr)
            continue
        in_channels = int(getattr(pipe.unet.config, "in_channels", -1))
        load_s = round(time.time() - t_load, 1)
        print(f"  loaded in {load_s}s — UNet in_channels={in_channels} "
              f"({'true 9-ch inpaint' if in_channels == 9 else 'legacy mask-blend'})", flush=True)

        for strength in strengths:
            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            t0 = time.time()
            try:
                result = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    image=source,
                    mask_image=mask,
                    width=size,
                    height=size,
                    strength=strength,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                ).images[0]
            except Exception as exc:  # noqa: BLE001
                summary["runs"].append(
                    {"checkpoint": ckpt_label, "strength": strength, "error": str(exc)}
                )
                print(f"  strength {strength} FAILED: {exc}", file=sys.stderr)
                continue
            elapsed = time.time() - t0
            name = f"{ckpt_label}_s{strength}.png"
            result.save(out / name)
            stats = _preservation_stats(source, result, mask)
            run = {
                "checkpoint": ckpt_label,
                "unetInChannels": in_channels,
                "strength": strength,
                "seconds": round(elapsed, 1),
                "secPerStep": round(elapsed / max(args.steps, 1), 2),
                "peakMemGb": _peak_mem_gb(device),
                "output": name,
                **stats,
            }
            summary["runs"].append(run)
            print(
                f"  strength {strength}: {run['seconds']}s "
                f"({run['secPerStep']}s/step) | inside Δ={stats['insideMeanDiff']} "
                f"outside Δ={stats['outsideMeanDiff']} | peak={run['peakMemGb']}GB",
                flush=True,
            )

        del pipe
        if device == "mps":
            torch.mps.empty_cache()

    # GO heuristic: at least one run where the masked region changed a lot (inside Δ high)
    # while outside stayed ~stable (outside Δ low) — i.e. a real localized edit.
    good = [
        r for r in summary["runs"]
        if "error" not in r and r["insideMeanDiff"] >= 15 and r["outsideMeanDiff"] <= 8
    ]
    summary["verdict"] = "GO" if good else "REVIEW"
    summary["bestRuns"] = sorted(
        good, key=lambda r: (r["outsideMeanDiff"], -r["insideMeanDiff"])
    )[:3]

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"VERDICT: {summary['verdict']}  (outputs + summary.json in {out})")
    print("GO criterion: inside-mask Δ ≥ 15 AND outside-mask Δ ≤ 8 (localized edit, "
          "outside preserved). Eyeball the PNGs — seams should blend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
