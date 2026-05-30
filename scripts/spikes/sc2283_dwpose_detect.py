#!/usr/bin/env python3
"""sc-2283 spike, step 2 — DWPose whole-body detector (photo -> keypoints).

Runs in the DWPose detection venv (/Users/michael/.dwpose-spike, onnxruntime +
rtmlib). For each source image: run RTMPose whole-body (DWPose-l: dw-ll_ucoco_384),
convert COCO-WholeBody-133 -> the SceneWorks OpenPose layout (18 body + 21x2 hands +
68 face), normalise to [0,1], carry per-keypoint confidence + a per-person bbox +
source aspect, render a skeleton preview through the PRODUCTION renderer
(scene_worker.openpose_skeleton.draw_wholebody), and emit one pose record per
detected person.

This is the *producer* the epic needs; its output drops straight into the
*consumer* (draw_wholebody, PR #376) the Z-Image Fun-CN pose head was trained on.

rtmlib + RTMPose/DWPose are Apache-2.0; onnxruntime is the same stack already
shipped/validated for InstantID's antelopev2 (arm64, no source builds).

USAGE (from repo root, in the dwpose venv):
    /Users/michael/.dwpose-spike/venv/bin/python scripts/spikes/sc2283_dwpose_detect.py \
        --images "/tmp/sc2283/sources/src_*.png" --out /tmp/sc2283/poses \
        --device mps --min-conf 0.3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---- COCO-WholeBody-133 -> SceneWorks OpenPose-18 index map ------------------
# WholeBody order: 0-16 COCO body, 17-22 feet, 23-90 face(68), 91-111 left hand,
# 112-132 right hand. COCO body order: 0 nose,1 l_eye,2 r_eye,3 l_ear,4 r_ear,
# 5 l_sho,6 r_sho,7 l_elb,8 r_elb,9 l_wri,10 r_wri,11 l_hip,12 r_hip,13 l_kne,
# 14 r_kne,15 l_ank,16 r_ank. OpenPose-18 wants neck(1) = midpoint of shoulders.
COCO_TO_OPENPOSE = {
    0: 0,   # nose
    # 1 neck -> computed
    2: 6,   # r_sho
    3: 8,   # r_elb
    4: 10,  # r_wri
    5: 5,   # l_sho
    6: 7,   # l_elb
    7: 9,   # l_wri
    8: 12,  # r_hip
    9: 14,  # r_kne
    10: 16,  # r_ank
    11: 11,  # l_hip
    12: 13,  # l_kne
    13: 15,  # l_ank
    14: 2,  # r_eye
    15: 1,  # l_eye
    16: 4,  # r_ear
    17: 3,  # l_ear
}
FACE_SLICE = slice(23, 91)        # 68 points
LHAND_SLICE = slice(91, 112)      # 21 points
RHAND_SLICE = slice(112, 133)     # 21 points


def _pt(kps: np.ndarray, sc: np.ndarray, i: int, w: int, h: int) -> list:
    """[x_norm, y_norm, conf] for wholebody index i."""
    return [float(kps[i, 0]) / w, float(kps[i, 1]) / h, float(sc[i])]


def _seq(kps: np.ndarray, sc: np.ndarray, sl: slice, w: int, h: int) -> list:
    return [_pt(kps, sc, i, w, h) for i in range(sl.start, sl.stop)]


def wholebody_to_openpose(kps: np.ndarray, sc: np.ndarray, w: int, h: int) -> dict:
    """Convert one person's (133,2)+(133,) detection into the SceneWorks pose record."""
    body: list = [None] * 18
    for op_idx, coco_idx in COCO_TO_OPENPOSE.items():
        body[op_idx] = _pt(kps, sc, coco_idx, w, h)
    # neck = midpoint of shoulders (coco 5=l_sho, 6=r_sho); conf = min of the two
    ls, rs = kps[5], kps[6]
    body[1] = [float((ls[0] + rs[0]) / 2) / w, float((ls[1] + rs[1]) / 2) / h,
               float(min(sc[5], sc[6]))]
    left = _seq(kps, sc, LHAND_SLICE, w, h)
    right = _seq(kps, sc, RHAND_SLICE, w, h)
    face = _seq(kps, sc, FACE_SLICE, w, h)
    return {"keypoints": body, "hands": [left, right], "face": face}


def _facing(body: list, min_conf: float) -> str:
    def ok(i):
        return body[i] is not None and body[i][2] >= min_conf
    nose, r_eye, l_eye, r_ear, l_ear = ok(0), ok(14), ok(15), ok(16), ok(17)
    if not nose and not r_eye and not l_eye:
        return "back"
    if r_ear and l_ear:
        return "front"
    if r_ear != l_ear:  # only one ear -> turned
        return "profile"
    return "front"


def _bbox(*groups: list, min_conf: float) -> list | None:
    xs, ys = [], []
    for g in groups:
        for p in g:
            if p is not None and p[2] >= min_conf:
                xs.append(p[0]); ys.append(p[1])
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _mean_conf(group: list) -> float:
    cs = [p[2] for p in group if p is not None]
    return float(np.mean(cs)) if cs else 0.0


def _thresholded(group: list, min_conf: float) -> list:
    """Render-ready copy: zero out conf for sub-threshold points (renderer drops conf<=0)."""
    out = []
    for p in group:
        if p is None or p[2] < min_conf:
            out.append(None)
        else:
            out.append([p[0], p[1], p[2]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob of source images")
    ap.add_argument("--out", default="/tmp/sc2283/poses")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--mode", default="performance", choices=["performance", "balanced", "lightweight"])
    ap.add_argument("--min-conf", type=float, default=0.3, help="render/threshold conf cutoff")
    ap.add_argument("--render-size", type=int, default=768, help="skeleton preview width (height scaled to aspect)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # production renderer (numpy + cv2 only)
    sys.path.insert(0, str(Path("apps/worker").resolve()))
    from scene_worker.openpose_skeleton import draw_wholebody

    from rtmlib import Wholebody

    # Build the detector; fall back to CPU if the requested provider is unavailable.
    device = args.device
    t0 = time.time()
    try:
        model = Wholebody(mode=args.mode, backend="onnxruntime", device=device)
    except Exception as e:  # noqa: BLE001
        print(f"[detect] device={device} failed ({e!r}); falling back to cpu", flush=True)
        device = "cpu"
        model = Wholebody(mode=args.mode, backend="onnxruntime", device=device)
    print(f"[detect] Wholebody(mode={args.mode}, device={device}) ready in {time.time() - t0:.1f}s", flush=True)

    paths = sorted(glob.glob(args.images))
    if not paths:
        print(f"[detect] no images match {args.images}", flush=True)
        return 1

    index = []
    for path in paths:
        img = cv2.imread(path)  # BGR
        if img is None:
            print(f"[detect] SKIP unreadable {path}", flush=True)
            continue
        h, w = img.shape[:2]
        t1 = time.time()
        keypoints, scores = model(img)  # (N,133,2) px, (N,133)
        dt = (time.time() - t1) * 1000.0
        n = 0 if keypoints is None else len(keypoints)
        print(f"[detect] {Path(path).name}: {n} person(s) in {dt:.0f}ms ({w}x{h})", flush=True)

        poses = []
        order = []
        for i in range(n):
            rec = wholebody_to_openpose(keypoints[i], scores[i], w, h)
            bbox = _bbox(rec["keypoints"], rec["hands"][0], rec["hands"][1], rec["face"], min_conf=args.min_conf)
            area = 0.0 if bbox is None else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            order.append((area, i, rec, bbox))
        order.sort(key=lambda t: -t[0])  # largest person first

        stem = Path(path).stem
        rh = round(args.render_size * h / w)
        for person_index, (_area, _i, rec, bbox) in enumerate(order):
            body_t = _thresholded(rec["keypoints"], args.min_conf)
            hands_t = [_thresholded(rec["hands"][0], args.min_conf), _thresholded(rec["hands"][1], args.min_conf)]
            face_t = _thresholded(rec["face"], args.min_conf)
            stick = max(6, round(min(args.render_size, rh) * 0.012))
            skel = draw_wholebody(args.render_size, rh, body_t, hands_t, face_t, stickwidth=stick)
            skel_path = out / f"{stem}_p{person_index}_skel.png"
            cv2.imwrite(str(skel_path), cv2.cvtColor(skel, cv2.COLOR_RGB2BGR))
            # overlay on source for QA
            src_rs = cv2.resize(img, (args.render_size, rh))
            skel_bgr = cv2.cvtColor(skel, cv2.COLOR_RGB2BGR)
            mask = skel.any(axis=2)
            overlay = src_rs.copy()
            overlay[mask] = cv2.addWeighted(src_rs, 0.25, skel_bgr, 0.75, 0)[mask]
            cv2.imwrite(str(out / f"{stem}_p{person_index}_overlay.png"), overlay)

            poses.append({
                "personIndex": person_index,
                "bbox": bbox,
                "facing": _facing(rec["keypoints"], args.min_conf),
                "meanConf": {
                    "body": round(_mean_conf(rec["keypoints"]), 3),
                    "hands": round((_mean_conf(rec["hands"][0]) + _mean_conf(rec["hands"][1])) / 2, 3),
                    "face": round(_mean_conf(rec["face"]), 3),
                },
                "keypoints": rec["keypoints"],
                "hands": rec["hands"],
                "face": rec["face"],
                "skeletonPreview": str(skel_path),
            })

        record = {
            "source": Path(path).name,
            "sourcePath": str(Path(path).resolve()),
            "sourceWidth": w,
            "sourceHeight": h,
            "sourceAspect": round(w / h, 4),
            # rtmlib mode->model: performance=RTMW-x(cocktail14,384x288), balanced/lightweight differ.
            # All emit COCO-WholeBody-133, so the render path is identical regardless.
            "detector": f"rtmlib/wholebody-{args.mode}",
            "detectorBackend": "onnxruntime",
            "device": device,
            "detectMs": round(dt, 1),
            "minConf": args.min_conf,
            "poses": poses,
        }
        json_path = out / f"{stem}.json"
        json_path.write_text(json.dumps(record, indent=2))
        index.append({"source": record["source"], "json": str(json_path), "nPoses": len(poses),
                      "meanConf": [p["meanConf"] for p in poses]})
        print(f"[detect]   -> {json_path}  poses={len(poses)} "
              + " ".join(f"p{p['personIndex']}={p['facing']}/{p['meanConf']}" for p in poses), flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[detect] DONE  wrote {len(index)} source record(s) to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
