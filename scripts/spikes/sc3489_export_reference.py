"""sc-3489 spike: export Real-ESRGAN (RRDBNet) to ONNX + produce a torch reference.

Reproduces the EXACT RRDBNet from apps/worker/scene_worker/upscalers.py, loads the
shipped nateraw/real-esrgan weights (RealESRGAN_x2plus / x4plus), exports an ONNX
graph with dynamic H/W, and renders a tiled torch reference upscale that the Rust
`ort` spike (sc3489_ort_upscale) is compared against. Also runs onnxruntime (CPU)
inside Python on the same tiles to confirm the ONNX graph itself is faithful before
the Rust leg.

Run with the spike venv:
  /tmp/sc3489_venv/bin/python scripts/spikes/sc3489_export_reference.py \
      --factor 4 --image poses/standing_09.png --outdir /tmp/sc3489
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MODEL_SPECS = {
    2: {"repo": "nateraw/real-esrgan", "file": "RealESRGAN_x2plus.pth"},
    4: {"repo": "nateraw/real-esrgan", "file": "RealESRGAN_x4plus.pth"},
}

TILE_SIZE = 512
TILE_PAD = 16


# ---- RRDBNet (verbatim port of upscalers.py:_rrdbnet_class) ------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, *, num_in_ch, num_out_ch, scale, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_state_dict(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("params_ema", "params", "state_dict"):
            if isinstance(ckpt.get(key), dict):
                state = ckpt[key]
                break
    return {str(k).removeprefix("module."): v for k, v in state.items()}


def infer_blocks(state: dict) -> int:
    idxs = [int(m.group(1)) for k in state if (m := re.match(r"body\.(\d+)\.", k))]
    return max(idxs) + 1 if idxs else 23


def build_model(factor: int, weights: Path) -> RRDBNet:
    state = load_state_dict(weights)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=factor, num_block=infer_blocks(state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing {len(missing)} tensors"
    assert not unexpected, f"unexpected {len(unexpected)} tensors"
    model.eval()
    return model


def tile_slices(w: int, h: int, tile: int):
    if tile <= 0 or tile >= max(w, h):
        return [(0, 0, w, h)]
    out = []
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            out.append((x0, y0, min(x0 + tile, w), min(y0 + tile, h)))
    return out


def run_tiled(infer, image: Image.Image, factor: int) -> Image.Image:
    """infer: (np float32 HWC 0..1) -> np float32 HWC 0..1 at factor scale."""
    w, h = image.size
    out = np.zeros((h * factor, w * factor, 3), dtype=np.float32)
    for x0, y0, x1, y1 in tile_slices(w, h, TILE_SIZE):
        cx0, cy0 = max(0, x0 - TILE_PAD), max(0, y0 - TILE_PAD)
        cx1, cy1 = min(w, x1 + TILE_PAD), min(h, y1 + TILE_PAD)
        crop = image.crop((cx0, cy0, cx1, cy1)).convert("RGB")
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        res = infer(arr)
        ix0, iy0 = (x0 - cx0) * factor, (y0 - cy0) * factor
        ix1, iy1 = ix0 + (x1 - x0) * factor, iy0 + (y1 - y0) * factor
        out[y0 * factor : y1 * factor, x0 * factor : x1 * factor, :] = res[iy0:iy1, ix0:ix1, :]
    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, choices=[2, 4], required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    factor = args.factor

    from huggingface_hub import hf_hub_download

    spec = MODEL_SPECS[factor]
    weights = Path(hf_hub_download(spec["repo"], spec["file"]))
    print(f"[weights] {weights}")

    model = build_model(factor, weights)

    # ---- ONNX export (dynamic H/W) ----
    onnx_path = outdir / f"real_esrgan_x{factor}.onnx"
    dummy = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "H", 3: "W"}},
        dynamo=False,
    )
    print(f"[onnx]    {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    image = Image.open(args.image).convert("RGB")
    print(f"[input]   {args.image} {image.size}")

    # ---- torch reference (tiled) ----
    def torch_infer(arr):
        with torch.inference_mode():
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            r = model(t)
            return r.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().numpy()

    ref = run_tiled(torch_infer, image, factor)
    ref_path = outdir / f"ref_torch_x{factor}.png"
    ref.save(ref_path)
    print(f"[torch]   {ref_path} {ref.size}")

    # ---- onnxruntime (CPU) sanity on the SAME tiling ----
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def ort_infer(arr):
        t = arr.transpose(2, 0, 1)[None].astype(np.float32)
        r = sess.run(["output"], {"input": t})[0]
        return r[0].transpose(1, 2, 0)

    ort_img = run_tiled(ort_infer, image, factor)
    ort_path = outdir / f"ref_ortcpu_x{factor}.png"
    ort_img.save(ort_path)

    a = np.asarray(ref, dtype=np.int16)
    b = np.asarray(ort_img, dtype=np.int16)
    diff = np.abs(a - b)
    mse = float((diff.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10((255.0**2) / mse)
    print(f"[ort-cpu] {ort_path}")
    print(f"[parity torch-vs-ortcpu] max|Δ|={int(diff.max())} mean|Δ|={diff.mean():.4f} PSNR={psnr:.2f}dB")

    # also save the input copy at known path for the Rust leg
    in_copy = outdir / "input.png"
    image.save(in_copy)
    print(f"[input-copy] {in_copy}")


if __name__ == "__main__":
    main()
