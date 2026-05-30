#!/usr/bin/env python3
"""sc-2283 spike, step 3 — the lock A/B (body-only vs full DWPose).

Runs in the mflux SIDECAR venv. For each detected pose (from
sc2283_dwpose_detect.py), render TWO control images at the source's native aspect:
  - body-only  : draw_bodypose(...)          (what ships today)
  - whole-body : draw_wholebody(..., hands, face)  (full DWPose, PR #376 renderer)
Then generate a NEW generic character into each, at a shared seed/prompt, across a
scale sweep, so the ONLY variable is whether hands/face are in the control. Probes
the sc-2257 nonlinearity (lock @0.9-1.0, drift higher) to see if hands/face firm it.

USAGE (repo root, sidecar venv):
    /Users/michael/mlx-flux-venv/bin/python scripts/spikes/sc2283_lock_ab.py \
        --cn /path/Fun-Controlnet-Union...safetensors --poses-dir /tmp/sc2283/poses \
        --out /tmp/sc2283/ab --scales 0.65,0.9,1.3 --steps 8 --seed 1234
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

NEW_SUBJECT = ("a full body studio photograph of a person in a plain white t-shirt and blue jeans, "
               "plain light grey seamless background, soft studio lighting, sharp focus, photorealistic, 50mm")


def _thresh(group, min_conf):
    out = []
    for p in group:
        out.append(None if (p is None or p[2] < min_conf) else (float(p[0]), float(p[1])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cn", required=True)
    ap.add_argument("--poses-dir", default="/tmp/sc2283/poses")
    ap.add_argument("--out", default="/tmp/sc2283/ab")
    ap.add_argument("--prompt", default=NEW_SUBJECT)
    ap.add_argument("--scales", default="0.65,0.9,1.3")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--person", type=int, default=0, help="which detected person (0=largest)")
    args = ap.parse_args()

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path("apps/worker").resolve()))
    from scene_worker.openpose_skeleton import draw_bodypose, draw_wholebody

    from mflux.models.z_image.variants.z_image_control import ZImageControl

    pose_files = sorted(p for p in Path(args.poses_dir).glob("*.json") if p.name != "index.json")
    if not pose_files:
        print(f"[ab] no pose JSONs in {args.poses_dir}", flush=True)
        return 1

    t0 = time.time()
    model = ZImageControl(control_weights_path=args.cn)
    print(f"[ab] ZImageControl loaded in {time.time() - t0:.1f}s (bits={model.bits})", flush=True)

    manifest = []
    for pf in pose_files:
        rec = json.loads(pf.read_text())
        if not rec["poses"]:
            print(f"[ab] {rec['source']}: no poses, skip", flush=True)
            continue
        pose = next((p for p in rec["poses"] if p["personIndex"] == args.person), rec["poses"][0])
        W, H = rec["sourceWidth"], rec["sourceHeight"]
        stem = pf.stem
        d = out_root / stem
        d.mkdir(parents=True, exist_ok=True)

        body = _thresh(pose["keypoints"], args.min_conf)
        hands = [_thresh(pose["hands"][0], args.min_conf), _thresh(pose["hands"][1], args.min_conf)]
        face = _thresh(pose["face"], args.min_conf)
        stick = max(6, round(min(W, H) * 0.012))

        ctrl_body = draw_bodypose(W, H, body, stickwidth=stick)
        ctrl_whole = draw_wholebody(W, H, body, hands, face, stickwidth=stick)
        body_path = d / "control_body.png"
        whole_path = d / "control_whole.png"
        Image.fromarray(ctrl_body).save(body_path)
        Image.fromarray(ctrl_whole).save(whole_path)
        print(f"[ab] {stem}: {W}x{H} facing={pose['facing']} conf={pose['meanConf']}", flush=True)

        for kind, ctrl_path in (("body", body_path), ("whole", whole_path)):
            for sc in scales:
                t1 = time.time()
                img = model.generate_image(
                    seed=args.seed,
                    prompt=args.prompt,
                    control_image_path=str(ctrl_path),
                    control_context_scale=sc,
                    num_inference_steps=args.steps,
                    height=H,
                    width=W,
                )
                dst = d / f"{kind}_s{sc}.png"
                getattr(img, "image", img).save(dst)
                manifest.append({"stem": stem, "kind": kind, "scale": sc, "out": str(dst),
                                 "control": str(ctrl_path), "W": W, "H": H})
                print(f"[ab]   {kind:5} scale={sc:<4} -> {dst.name}  ({time.time() - t1:.1f}s)", flush=True)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ab] DONE  {len(manifest)} generations -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
