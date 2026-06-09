"""sc-3635 spike: SAM2 box-prompt segmentation reference + ONNX/CoreML validation.

Mirrors the production `_Sam2Segmenter` contract (apps/worker/scene_worker/
person_adapters.py): box prompt -> single best mask (multimask_output semantics:
argmax of IoU scores) -> binary "L" mask. This script establishes three things the
GO/NO-GO decision rests on:

  1. PyTorch baseline  -- transformers Sam2Model on facebook/sam2.1-hiera-large
     (a faithful reimplementation of the same official weights the worker ships;
     SAM2.1-large is the same architecture as the v1 `sam2_hiera_large.pt` baseline,
     an incremental quality improvement). This is the quality source-of-truth.
  2. ONNX faithfulness -- onnxruntime CPU on the community two-graph image-predictor
     split (onnx-community/sam2.1-hiera-large-ONNX: vision_encoder.onnx +
     prompt_encoder_mask_decoder.onnx). IoU vs the PyTorch baseline.
  3. CoreML EP         -- onnxruntime CoreMLExecutionProvider (falls back to CPU for
     unsupported ops). IoU vs CPU + latency + per-op provider placement (the real
     risk: Hiera encoder op coverage). The Rust `ort` leg (sc3635_ort_sam2) re-runs
     the exact same graphs and must match these masks.

Run with the spike venv (~/mlx-flux-venv has torch+MPS, transformers, onnx, ort):
  ~/mlx-flux-venv/bin/python scripts/spikes/sc3635_reference.py \
      --image /tmp/sc3635/zidane.jpg --box 710,40 1150,715 --outdir /tmp/sc3635/zidane
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

REPO = "onnx-community/sam2.1-hiera-large-ONNX"
PT_MODEL = "facebook/sam2.1-hiera-large"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 1024


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 127
    b = b > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def preprocess(image: Image.Image) -> np.ndarray:
    """SAM2 preprocessing: resize to 1024x1024 (square stretch, bilinear), /255,
    ImageNet normalize -> float32 NCHW. Matches Sam2ImageProcessorFast."""
    im = image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)  # [1,3,1024,1024]


def box_to_1024(box, w, h):
    x1, y1, x2, y2 = box
    sx, sy = SIZE / w, SIZE / h
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def upsample_logits_to_L(low_res_logits: np.ndarray, w: int, h: int) -> np.ndarray:
    """low_res_logits: [Hm,Wm] mask logits -> binary uint8 L mask at original (w,h)
    via bilinear upsample of the logits then threshold>0 (SAM2 post-process; the
    preprocessing was a pure stretch to 1024 so no letterbox un-pad is needed)."""
    t = torch.from_numpy(np.ascontiguousarray(low_res_logits))[None, None].float()
    up = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return (up[0, 0].numpy() > 0).astype(np.uint8) * 255


# --------------------------------------------------------------------------- #
# PyTorch baseline (transformers Sam2Model, processor bypassed)               #
# --------------------------------------------------------------------------- #
def pytorch_baseline(pix: np.ndarray, box1024, outdir: Path, w: int, h: int):
    from transformers import Sam2Model

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Sam2Model.from_pretrained(PT_MODEL).to(device).eval()

    pix_t = torch.from_numpy(pix).to(device)
    box_t = torch.tensor([[box1024]], dtype=torch.float32, device=device)  # [1,1,4]
    t = time.time()
    with torch.inference_mode():
        out = model(pixel_values=pix_t, input_boxes=box_t, multimask_output=True)
    dt = time.time() - t
    pred = out.pred_masks.float().cpu().numpy()  # [1,1,3,Hm,Wm]
    scores = out.iou_scores.float().cpu().numpy()[0, 0]  # [3]
    best = int(np.argmax(scores))
    mask = upsample_logits_to_L(pred[0, 0, best], w, h)
    Image.fromarray(mask, mode="L").save(outdir / "ref_pt_mask.png")
    print(f"[pytorch] device={device} {dt*1000:.0f}ms scores={scores.round(3).tolist()} "
          f"best={best} pred_masks_shape={pred.shape}")
    return mask


# --------------------------------------------------------------------------- #
# ONNX runtime (CPU + CoreML)                                                  #
# --------------------------------------------------------------------------- #
def inline_model(src: str, dst: str) -> str:
    """Re-save an external-data ONNX as a single self-contained file. The CoreML EP
    cannot resolve `.onnx_data` external weights when it compiles sub-graphs
    ('model_path must not be empty'); inlining is the standard fix and is also how
    the real port would provision weights. Cached on disk."""
    import os
    import onnx

    import shutil

    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # onnx refuses to follow the HF-cache symlink for external data: stage real
    # copies (.onnx + .onnx_data) in a tmp dir, then load+inline.
    stage = dst + ".stage"
    os.makedirs(stage, exist_ok=True)
    base = os.path.basename(src)
    shutil.copy(os.path.realpath(src), os.path.join(stage, base))
    shutil.copy(os.path.realpath(src + "_data"), os.path.join(stage, base + "_data"))
    m = onnx.load(os.path.join(stage, base), load_external_data=True)
    onnx.save_model(m, dst, save_as_external_data=False)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[inline] {src} -> {dst} ({os.path.getsize(dst)/1e6:.0f} MB)")
    return dst


def make_session(path: str, provider: str, profile: bool = False):
    so = ort.SessionOptions()
    if profile:
        so.enable_profiling = True
    providers = (
        [("CoreMLExecutionProvider", {}), "CPUExecutionProvider"]
        if provider == "coreml"
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def op_placement(session, base_msg: str):
    """Parse the profiling JSON for per-node EP placement."""
    prof = session.end_profiling()
    data = json.load(open(prof))
    by_provider = Counter()
    for e in data:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            ep = e.get("args", {}).get("provider", "?")
            by_provider[ep] += 1
    total = sum(by_provider.values())
    print(f"[op-placement {base_msg}] {dict(by_provider)} (total kernels={total})")
    return dict(by_provider), total


def run_onnx(enc_path, dec_path, pix, box1024, provider, outdir, w, h, tag):
    enc = make_session(enc_path, provider, profile=True)
    dec = make_session(dec_path, provider, profile=True)

    # encoder
    t = time.time()
    e0, e1, e2 = enc.run(None, {"pixel_values": pix})
    enc_dt = time.time() - t

    # decoder: box prompt. Graph wants input_points/input_labels/input_boxes.
    # A box-only prompt: one dummy padding point (label -10 = ignored by the
    # prompt encoder, per HF Sam2) + the dedicated input_boxes path.
    input_points = np.zeros((1, 1, 1, 2), dtype=np.float32)
    input_labels = np.full((1, 1, 1), -10, dtype=np.int64)  # -10 = padding/ignored
    input_boxes = np.array([[box1024]], dtype=np.float32)  # [1,1,4]
    feeds = {
        "input_points": input_points,
        "input_labels": input_labels,
        "input_boxes": input_boxes,
        "image_embeddings.0": e0,
        "image_embeddings.1": e1,
        "image_embeddings.2": e2,
    }
    t = time.time()
    iou_scores, pred_masks, _ = dec.run(None, feeds)
    dec_dt = time.time() - t

    scores = iou_scores[0, 0]  # [num_masks]
    best = int(np.argmax(scores))
    low = pred_masks[0, 0, best]  # [Hm,Wm] logits
    mask_arr = upsample_logits_to_L(low, w, h)
    Image.fromarray(mask_arr, mode="L").save(outdir / f"mask_{tag}.png")

    enc_ph, enc_tot = op_placement(enc, f"{tag}/encoder")
    dec_ph, dec_tot = op_placement(dec, f"{tag}/decoder")
    print(
        f"[onnx {tag}] enc={enc_dt*1000:.0f}ms dec={dec_dt*1000:.0f}ms "
        f"total={(enc_dt+dec_dt)*1000:.0f}ms scores={scores.round(3).tolist()} best={best} "
        f"pred_masks_shape={pred_masks.shape}"
    )
    return mask_arr, {
        "enc_ms": enc_dt * 1000,
        "dec_ms": dec_dt * 1000,
        "enc_placement": enc_ph,
        "dec_placement": dec_ph,
        "scores": scores.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--box", nargs=2, required=True, help="x1,y1 x2,y2")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repeat", type=int, default=3, help="timing repeats for ORT")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (x1, y1) = (float(v) for v in args.box[0].split(","))
    (x2, y2) = (float(v) for v in args.box[1].split(","))
    box = [x1, y1, x2, y2]

    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    box1024 = box_to_1024(box, w, h)
    print(f"[input] {args.image} {image.size} box={box} -> box1024={[round(v,1) for v in box1024]}")

    from huggingface_hub import hf_hub_download

    enc_raw = hf_hub_download(REPO, "onnx/vision_encoder.onnx")
    hf_hub_download(REPO, "onnx/vision_encoder.onnx_data")
    dec_raw = hf_hub_download(REPO, "onnx/prompt_encoder_mask_decoder.onnx")
    hf_hub_download(REPO, "onnx/prompt_encoder_mask_decoder.onnx_data")

    # Inline external weights into single-file graphs (shared cache, CoreML-safe).
    inl = "/tmp/sc3635/onnx_inlined"
    enc_path = inline_model(enc_raw, f"{inl}/vision_encoder.onnx")
    dec_path = inline_model(dec_raw, f"{inl}/prompt_encoder_mask_decoder.onnx")

    pix = preprocess(image)

    # Dump identical inputs for the Rust leg.
    pix.tofile(outdir / "pixel_values.f32")  # [1,3,1024,1024] row-major
    json.dump(
        {"box_orig": box, "box1024": box1024, "orig_w": w, "orig_h": h,
         "enc_path": enc_path, "dec_path": dec_path,
         "input_labels_pad": -10},
        open(outdir / "inputs.json", "w"), indent=1,
    )
    image.save(outdir / "input.png")

    pt_mask = pytorch_baseline(pix, box1024, outdir, w, h)

    cpu_mask, cpu_stats = run_onnx(enc_path, dec_path, pix, box1024, "cpu", outdir, w, h, "ortcpu")
    cml_mask, cml_stats = run_onnx(enc_path, dec_path, pix, box1024, "coreml", outdir, w, h, "ortcoreml")

    # timing repeats (steady-state, drop first compile pass already done above)
    def time_provider(provider):
        enc = make_session(enc_path, provider)
        dec = make_session(dec_path, provider)
        ip = np.zeros((1, 1, 1, 2), dtype=np.float32)
        il = np.full((1, 1, 1), -1, dtype=np.int64)
        ib = np.array([[box1024]], dtype=np.float32)
        es, ds = [], []
        for _ in range(args.repeat):
            t = time.time(); e0, e1, e2 = enc.run(None, {"pixel_values": pix}); es.append(time.time() - t)
            feeds = {"input_points": ip, "input_labels": il, "input_boxes": ib,
                     "image_embeddings.0": e0, "image_embeddings.1": e1, "image_embeddings.2": e2}
            t = time.time(); dec.run(None, feeds); ds.append(time.time() - t)
        return min(es) * 1000, min(ds) * 1000

    cpu_enc_ms, cpu_dec_ms = time_provider("cpu")
    cml_enc_ms, cml_dec_ms = time_provider("coreml")

    print("\n================ SUMMARY ================")
    print(f"IoU  ortcpu    vs pytorch : {iou(cpu_mask, pt_mask):.4f}")
    print(f"IoU  ortcoreml vs pytorch : {iou(cml_mask, pt_mask):.4f}")
    print(f"IoU  ortcoreml vs ortcpu  : {iou(cml_mask, cpu_mask):.4f}")
    print(f"latency CPU    : enc {cpu_enc_ms:.0f}ms + dec {cpu_dec_ms:.0f}ms = {cpu_enc_ms+cpu_dec_ms:.0f}ms/frame")
    print(f"latency CoreML : enc {cml_enc_ms:.0f}ms + dec {cml_dec_ms:.0f}ms = {cml_enc_ms+cml_dec_ms:.0f}ms/frame")

    summary = {
        "image": args.image, "box": box, "size": [w, h],
        "iou_ortcpu_vs_pt": iou(cpu_mask, pt_mask),
        "iou_ortcoreml_vs_pt": iou(cml_mask, pt_mask),
        "iou_ortcoreml_vs_ortcpu": iou(cml_mask, cpu_mask),
        "cpu_stats": cpu_stats, "coreml_stats": cml_stats,
        "latency_cpu_ms": {"enc": cpu_enc_ms, "dec": cpu_dec_ms},
        "latency_coreml_ms": {"enc": cml_enc_ms, "dec": cml_dec_ms},
    }
    json.dump(summary, open(outdir / "summary.json", "w"), indent=1)
    print(f"\n[written] {outdir}")


if __name__ == "__main__":
    main()
