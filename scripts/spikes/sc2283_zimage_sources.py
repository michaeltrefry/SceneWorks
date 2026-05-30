#!/usr/bin/env python3
"""sc-2283 spike, step 1 — synthesize base Z-Image full-body source photos.

Runs in the mflux SIDECAR venv (/Users/michael/mlx-flux-venv). Generates a few
diverse full-body person images with **base** Z-Image-Turbo (the Fun-CN loaded but
``control_context_scale=0.0`` over a black control image == base parity, per the
sc-2257 validate gate) to use as DWPose detection sources for the lock A/B. No
external photos needed.

USAGE:
    /Users/michael/mlx-flux-venv/bin/python scripts/spikes/sc2283_zimage_sources.py \
        --cn /path/to/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
        --out /tmp/sc2283/sources --width 768 --height 1280 --steps 8 --seed 7
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Full-body prompts chosen to exercise the hand/face lock test:
#  - arms_crossed: hands partly tucked (hard case)
#  - sitting:      hands resting on knees (clearly visible, relaxed)
#  - dance:        one arm raised, open hand (the strongest hand-lock probe)
PROMPTS = {
    "arms_crossed": "a full body studio photograph of a young woman standing, arms crossed over chest, "
                    "looking at the camera, natural hands, plain light grey seamless background, soft studio "
                    "lighting, sharp focus, photorealistic, 50mm",
    "sitting": "a full body studio photograph of a man sitting on a simple wooden stool, both hands resting "
               "on his knees, looking at the camera, plain light grey seamless background, soft studio "
               "lighting, sharp focus, photorealistic, 50mm",
    "dance": "a full body studio photograph of a woman in a dynamic dance pose, one arm raised high with an "
             "open hand, the other arm extended, looking up, plain light grey seamless background, soft studio "
             "lighting, sharp focus, photorealistic, 50mm",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cn", required=True, help="Fun-Controlnet-Union safetensors path")
    ap.add_argument("--out", default="/tmp/sc2283/sources")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from mflux.models.z_image.variants.z_image_control import ZImageControl

    # A black control image: VAE-encoded then multiplied by control_context_scale=0.0,
    # so it contributes nothing -> pure base Z-Image-Turbo output.
    black = out / "_black_control.png"
    Image.fromarray(np.zeros((args.height, args.width, 3), dtype=np.uint8)).save(black)

    t0 = time.time()
    model = ZImageControl(control_weights_path=args.cn)
    print(f"[sources] ZImageControl loaded in {time.time() - t0:.1f}s (bits={model.bits})", flush=True)

    for name, prompt in PROMPTS.items():
        t1 = time.time()
        img = model.generate_image(
            seed=args.seed,
            prompt=prompt,
            control_image_path=str(black),
            control_context_scale=0.0,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
        )
        dst = out / f"src_{name}.png"
        pil = getattr(img, "image", img)
        pil.save(dst)
        print(f"[sources] {name:12} -> {dst}  ({time.time() - t1:.1f}s)  {pil.size}", flush=True)

    print("[sources] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
