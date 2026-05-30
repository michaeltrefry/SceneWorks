#!/usr/bin/env python3
"""sc-2283 spike, step 4 — objective adherence scoring + visual contact sheets.

Runs in the DWPose detection venv. Re-detects DWPose on every A/B output and
measures how well the generated image followed the *intended* pose:
  - body_err : mean normalized L2 between output body-18 and the intended body-18
               (same intended body drives both body-only and whole-body controls,
                so this shows whether adding hands/face HURTS the body lock).
  - hand_err / face_err : vs the intended hands/face. For body-only the hands are
               un-controlled (the model invents them) -> expect large/sparse; for
               whole-body -> expect small. This is the number that justifies (or
               kills) the epic.
Also builds a per-source montage [source | body-only sweep | whole-body sweep] for
human judgement of hand/finger/face fidelity (mangling isn't fully captured by L2).

USAGE (repo root, dwpose venv):
    /Users/michael/.dwpose-spike/venv/bin/python scripts/spikes/sc2283_score.py \
        --ab /tmp/sc2283/ab --poses-dir /tmp/sc2283/poses --device mps --min-conf 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc2283_dwpose_detect import wholebody_to_openpose  # noqa: E402


def _thresh(group, min_conf):
    return [None if (p is None or p[2] < min_conf) else (float(p[0]), float(p[1])) for p in group]


def _err(a, b):
    """mean normalized L2 over points present in both; (mean, n_matched)."""
    ds = []
    for pa, pb in zip(a, b):
        if pa is None or pb is None:
            continue
        ds.append(((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5)
    return (float(np.mean(ds)) if ds else None, len(ds))


def detect_one(model, img_bgr, min_conf):
    h, w = img_bgr.shape[:2]
    kps, sc = model(img_bgr)
    if kps is None or len(kps) == 0:
        return None
    # largest person by keypoint spread
    best, best_area = 0, -1.0
    for i in range(len(kps)):
        xs, ys = kps[i][:, 0], kps[i][:, 1]
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        if area > best_area:
            best, best_area = i, area
    rec = wholebody_to_openpose(kps[best], sc[best], w, h)
    return {
        "body": _thresh(rec["keypoints"], min_conf),
        "lhand": _thresh(rec["hands"][0], min_conf),
        "rhand": _thresh(rec["hands"][1], min_conf),
        "face": _thresh(rec["face"], min_conf),
    }


def _label(img, text, h=28):
    bar = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _fit(img, th):
    w = round(img.shape[1] * th / img.shape[0])
    return cv2.resize(img, (w, th))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", default="/tmp/sc2283/ab")
    ap.add_argument("--poses-dir", default="/tmp/sc2283/poses")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--mode", default="performance")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--th", type=int, default=384, help="montage thumb height")
    args = ap.parse_args()

    from rtmlib import Wholebody
    try:
        model = Wholebody(mode=args.mode, backend="onnxruntime", device=args.device)
    except Exception:  # noqa: BLE001
        model = Wholebody(mode=args.mode, backend="onnxruntime", device="cpu")

    manifest = json.loads((Path(args.ab) / "manifest.json").read_text())
    # intended pose per stem
    intended = {}
    for pf in Path(args.poses_dir).glob("*.json"):
        if pf.name == "index.json":
            continue
        rec = json.loads(pf.read_text())
        if rec["poses"]:
            p = rec["poses"][0]
            intended[pf.stem] = {
                "body": _thresh(p["keypoints"], args.min_conf),
                "lhand": _thresh(p["hands"][0], args.min_conf),
                "rhand": _thresh(p["hands"][1], args.min_conf),
                "face": _thresh(p["face"], args.min_conf),
                "src": rec["sourcePath"],
            }

    rows = []
    by_stem = {}
    for m in manifest:
        stem = m["stem"]
        out = cv2.imread(m["out"])
        det = detect_one(model, out, args.min_conf)
        intd = intended[stem]
        if det is None:
            rows.append({**m, "body_err": None, "hand_err": None, "face_err": None,
                         "n_body": 0, "n_hand": 0, "n_face": 0})
            by_stem.setdefault(stem, {})[(m["kind"], m["scale"])] = m["out"]
            continue
        body_err, n_body = _err(det["body"], intd["body"])
        lh_err, nlh = _err(det["lhand"], intd["lhand"])
        rh_err, nrh = _err(det["rhand"], intd["rhand"])
        face_err, n_face = _err(det["face"], intd["face"])
        hand_vals = [v for v in (lh_err, rh_err) if v is not None]
        hand_err = float(np.mean(hand_vals)) if hand_vals else None
        rows.append({**m, "body_err": body_err, "hand_err": hand_err, "face_err": face_err,
                     "n_body": n_body, "n_hand": nlh + nrh, "n_face": n_face})
        by_stem.setdefault(stem, {})[(m["kind"], m["scale"])] = m["out"]

    # ---- table ----
    scales = sorted({m["scale"] for m in manifest})
    print("\n=== sc-2283 lock A/B — adherence (mean normalized L2, lower=tighter; n=matched kps) ===")
    hdr = f"{'source':16} {'ctrl':6} {'scale':6} {'body_err':9} {'hand_err':9} {'face_err':9} {'n_hand':7} {'n_face':7}"
    print(hdr); print("-" * len(hdr))
    for stem in sorted(by_stem):
        for kind in ("body", "whole"):
            for sc in scales:
                r = next((x for x in rows if x["stem"] == stem and x["kind"] == kind and x["scale"] == sc), None)
                if not r:
                    continue
                def f(v):
                    return f"{v:.4f}" if isinstance(v, float) else "  n/a  "
                print(f"{stem:16} {kind:6} {sc:<6} {f(r['body_err']):9} {f(r['hand_err']):9} "
                      f"{f(r['face_err']):9} {r['n_hand']:<7} {r['n_face']:<7}")
        print()

    # ---- per-source montage ----
    th = args.th
    for stem, cells in by_stem.items():
        src = cv2.imread(intended[stem]["src"])
        cb = cv2.imread(str(Path(args.ab) / stem / "control_body.png"))
        cw = cv2.imread(str(Path(args.ab) / stem / "control_whole.png"))
        header = np.hstack([_label(_fit(src, th), "SOURCE"),
                            _label(_fit(cb, th), "ctrl body-only"),
                            _label(_fit(cw, th), "ctrl whole-body")])
        body_row = [_label(_fit(cv2.imread(cells[("body", sc)]), th), f"body s{sc}") for sc in scales if ("body", sc) in cells]
        whole_row = [_label(_fit(cv2.imread(cells[("whole", sc)]), th), f"whole s{sc}") for sc in scales if ("whole", sc) in cells]
        body_strip = np.hstack(body_row) if body_row else None
        whole_strip = np.hstack(whole_row) if whole_row else None

        # pad widths to align
        widths = [x.shape[1] for x in (header, body_strip, whole_strip) if x is not None]
        wmax = max(widths)
        def pad(x):
            if x is None or x.shape[1] == wmax:
                return x
            return np.hstack([x, np.zeros((x.shape[0], wmax - x.shape[1], 3), dtype=np.uint8)])
        stacked = np.vstack([pad(s) for s in (header, body_strip, whole_strip) if s is not None])
        mpath = Path(args.ab) / f"{stem}_montage.png"
        cv2.imwrite(str(mpath), stacked)
        print(f"[score] montage -> {mpath}")

    (Path(args.ab) / "scores.json").write_text(json.dumps(rows, indent=2))
    print(f"[score] DONE -> {Path(args.ab) / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
