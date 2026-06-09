"""sc-3635 spike: SAM2 variant sweep — encoder EP/precision/model-size tradeoff.

The reference run (sc3635_reference.py) showed fp32 CoreML is ~2.8x SLOWER than CPU
for the hiera-large encoder (ViT fragments into 50+ CoreML partitions). This sweep
answers the two questions that decide the EP + model recommendation:
  1. Can fp16 on CoreML (ANE) ever beat fp32 CPU for the encoder?
  2. What is the latency/quality tradeoff of smaller SAM2.1 models (base-plus, tiny)?

Encoder latency dominates (decoder is ~10ms), so we time the encoder across variants
and measure full-pipeline mask IoU vs the hiera-large fp32 CPU reference mask.

  ~/mlx-flux-venv/bin/python scripts/spikes/sc3635_variants.py --indir /tmp/sc3635/zidane
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

MODELS = {
    "large": "onnx-community/sam2.1-hiera-large-ONNX",
    "base-plus": "onnx-community/sam2.1-hiera-base-plus-ONNX",
    "tiny": "onnx-community/sam2.1-hiera-tiny-ONNX",
}


def stage_real(src: str, dst: str) -> str:
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    stage = dst + ".stage"
    os.makedirs(stage, exist_ok=True)
    base = os.path.basename(src)
    shutil.copy(os.path.realpath(src), os.path.join(stage, base))
    dpath = src + "_data"
    if os.path.exists(dpath):
        shutil.copy(os.path.realpath(dpath), os.path.join(stage, base + "_data"))
        m = onnx.load(os.path.join(stage, base), load_external_data=True)
        onnx.save_model(m, dst, save_as_external_data=False)
    else:
        shutil.copy(os.path.realpath(src), dst)
    shutil.rmtree(stage, ignore_errors=True)
    return dst


def fetch(repo: str, fname: str) -> str:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo, fname)
    dp = fname + "_data"
    try:
        hf_hub_download(repo, dp)
    except Exception:
        pass
    tag = repo.split("/")[-1] + "__" + os.path.basename(fname)
    return stage_real(p, f"/tmp/sc3635/variants/{tag}")


def session(path: str, provider: str):
    providers = (
        [("CoreMLExecutionProvider", {}), "CPUExecutionProvider"]
        if provider == "coreml"
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(path, providers=providers)


def upsample(low, w, h):
    import torch

    t = torch.from_numpy(np.ascontiguousarray(low))[None, None].float()
    up = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return (up[0, 0].numpy() > 0).astype(np.uint8) * 255


def iou(a, b):
    a, b = a > 127, b > 127
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum()) / float(u)


def time_encoder(sess, pix, n=4):
    itype = sess.get_inputs()[0].type  # tensor(float) or tensor(float16)
    px = pix.astype(np.float16) if "float16" in itype else pix.astype(np.float32)
    name = sess.get_inputs()[0].name
    ts = []
    out = None
    for _ in range(n):
        t = time.time()
        out = sess.run(None, {name: px})
        ts.append(time.time() - t)
    return min(ts) * 1000, out


def run_decoder(dec, emb, box1024):
    # decoder always fp32 CPU (tiny + CoreML-hostile); cast embeddings to fp32
    e = [x.astype(np.float32) for x in emb]
    feeds = {
        "input_points": np.zeros((1, 1, 1, 2), np.float32),
        "input_labels": np.full((1, 1, 1), -10, np.int64),
        "input_boxes": np.array([[box1024]], np.float32),
        "image_embeddings.0": e[0],
        "image_embeddings.1": e[1],
        "image_embeddings.2": e[2],
    }
    iou_scores, pred_masks, _ = dec.run(None, feeds)
    best = int(np.argmax(iou_scores[0, 0]))
    return pred_masks[0, 0, best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    args = ap.parse_args()
    indir = args.indir
    meta = json.load(open(os.path.join(indir, "inputs.json")))
    w, h, box1024 = meta["orig_w"], meta["orig_h"], meta["box1024"]
    pix = np.fromfile(os.path.join(indir, "pixel_values.f32"), dtype=np.float32).reshape(1, 3, 1024, 1024)
    ref = np.asarray(Image.open(os.path.join(indir, "mask_ortcpu.png")).convert("L"))

    variants = [
        ("large", "onnx/vision_encoder.onnx", "cpu"),
        ("large", "onnx/vision_encoder.onnx", "coreml"),
        ("large", "onnx/vision_encoder_fp16.onnx", "coreml"),
        ("large", "onnx/vision_encoder_fp16.onnx", "cpu"),
        ("base-plus", "onnx/vision_encoder.onnx", "cpu"),
        ("base-plus", "onnx/vision_encoder.onnx", "coreml"),
        ("base-plus", "onnx/vision_encoder_fp16.onnx", "coreml"),
        ("tiny", "onnx/vision_encoder.onnx", "cpu"),
        ("tiny", "onnx/vision_encoder_fp16.onnx", "coreml"),
    ]
    # decoders (fp32) per model size, cached
    decoders = {}
    rows = []
    for size, encfile, prov in variants:
        repo = MODELS[size]
        try:
            encp = fetch(repo, encfile)
        except Exception as e:
            print(f"[skip] {size} {encfile} {prov}: fetch failed {e}")
            continue
        try:
            sess = session(encp, prov)
            enc_ms, emb = time_encoder(sess, pix)
            if size not in decoders:
                decoders[size] = session(fetch(repo, "onnx/prompt_encoder_mask_decoder.onnx"), "cpu")
            low = run_decoder(decoders[size], emb, box1024)
            m = upsample(low, w, h)
            q = iou(m, ref)
            prec = "fp16" if "fp16" in encfile else "fp32"
            rows.append((size, prec, prov, enc_ms, q))
            print(f"[{size:9s} {prec} {prov:6s}] enc={enc_ms:7.0f}ms  IoU_vs_large_cpu={q:.4f}")
        except Exception as e:
            print(f"[FAIL] {size} {encfile} {prov}: {type(e).__name__}: {str(e)[:160]}")

    print("\n==== ENCODER LATENCY / QUALITY (IoU vs hiera-large fp32 CPU) ====")
    print(f"{'model':10s} {'prec':4s} {'EP':6s} {'enc_ms':>8s} {'IoU':>7s}")
    for size, prec, prov, ms, q in rows:
        print(f"{size:10s} {prec:4s} {prov:6s} {ms:8.0f} {q:7.4f}")
    json.dump([dict(model=s, prec=p, ep=e, enc_ms=m, iou=q) for s, p, e, m, q in rows],
              open(os.path.join(indir, "variants.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
