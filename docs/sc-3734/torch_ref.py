#!/usr/bin/env python
"""sc-3734 — faithful torch fp32 reference forward of the fused YOLO11m, built from the
SAME fused MLX weights the Rust path loads. Validates against the existing oracle
(refs.safetensors block4/10/16/19/22/final), then exports an EXTENDED oracle
(refs_ext.safetensors) with every intermediate backbone block (5,6,7,8,9) plus the
C2PSA sub-steps, so each Rust block can be isolated against clean ground-truth input.

Everything runs on CPU fp32, channel-major NCHW, no ultralytics dependency — the forward
is reassembled from the fused state-dict exactly as person_jobs.rs does.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
import safetensors.numpy as stn

CACHE = sys.argv[1] if len(sys.argv) > 1 else \
    f"{__import__('os').environ['HOME']}/Library/Application Support/SceneWorks/cache/person-detect"

W = {k: torch.from_numpy(v.astype(np.float32)) for k, v in
     stn.load_file(f"{CACHE}/yolo11m_fused_mlx.safetensors").items()}
ORACLE = stn.load_file(f"{CACHE}/refs.safetensors")

def wt(key):
    # fused weight is MLX layout (o,kh,kw,i) -> torch (o,i,kh,kw)
    return W[key].permute(0, 3, 1, 2).contiguous()

def conv(x, prefix, stride, pad, act=True):
    w = wt(f"{prefix}.conv.weight")
    b = W[f"{prefix}.conv.bias"]
    y = F.conv2d(x, w, b, stride=stride, padding=pad)
    return F.silu(y) if act else y

def conv_raw(x, prefix, stride, pad):
    w = wt(f"{prefix}.weight")
    b = W[f"{prefix}.bias"]
    return F.conv2d(x, w, b, stride=stride, padding=pad)

def depthwise3x3(x, prefix, act):
    w = wt(f"{prefix}.conv.weight")  # (C,1,3,3)
    b = W[f"{prefix}.conv.bias"]
    C = x.shape[1]
    y = F.conv2d(x, w, b, stride=1, padding=1, groups=C)
    return F.silu(y) if act else y

def bottleneck(x, prefix, shortcut):
    y = conv(x, f"{prefix}.cv1", 1, 1, True)
    y = conv(y, f"{prefix}.cv2", 1, 1, True)
    return x + y if shortcut else y

def c3k(x, prefix, shortcut):
    p1 = conv(x, f"{prefix}.cv1", 1, 0, True)
    p2 = conv(x, f"{prefix}.cv2", 1, 0, True)
    q = bottleneck(p1, f"{prefix}.m.0", shortcut)
    q = bottleneck(q, f"{prefix}.m.1", shortcut)
    cat = torch.cat([q, p2], dim=1)
    return conv(cat, f"{prefix}.cv3", 1, 0, True)

def c3k2(x, i, shortcut):
    cv1 = conv(x, f"model.{i}.cv1", 1, 0, True)
    a, b = cv1.chunk(2, dim=1)
    m = c3k(b, f"model.{i}.m.0", shortcut)
    cat = torch.cat([a, b, m], dim=1)
    return conv(cat, f"model.{i}.cv2", 1, 0, True)

def maxpool5(x):
    return F.max_pool2d(x, kernel_size=5, stride=1, padding=2)

def sppf(x, i, dump=None):
    y = conv(x, f"model.{i}.cv1", 1, 0, True)
    p1 = maxpool5(y)
    p2 = maxpool5(p1)
    p3 = maxpool5(p2)
    if dump is not None:
        dump["sppf_cv1"] = y
        dump["sppf_p1"] = p1
        dump["sppf_p2"] = p2
        dump["sppf_p3"] = p3
    cat = torch.cat([y, p1, p2, p3], dim=1)
    return conv(cat, f"model.{i}.cv2", 1, 0, True)

def attention(x, prefix, dump=None):
    B, C, H, Wd = x.shape
    N = H * Wd
    nh, kd, hd = 4, 32, 64
    qkv = conv(x, f"{prefix}.qkv", 1, 0, act=False)         # (B,512,H,W)
    q, k, v = qkv.view(B, nh, 2 * kd + hd, N).split([kd, kd, hd], dim=2)
    attn = (q.transpose(-2, -1) @ k) * (kd ** -0.5)          # (B,nh,N,N)
    attn = attn.softmax(dim=-1)
    x_attn = (v @ attn.transpose(-2, -1)).reshape(B, C, H, Wd)
    pe = depthwise3x3(v.reshape(B, C, H, Wd), f"{prefix}.pe", act=False)
    if dump is not None:
        dump["attn_xattn"] = x_attn
        dump["attn_pe"] = pe
    out = x_attn + pe
    return conv(out, f"{prefix}.proj", 1, 0, act=False)

def psablock(x, prefix, dump=None):
    a = attention(x, f"{prefix}.attn", dump)
    b1 = x + a
    if dump is not None:
        dump["psa_after_attn"] = b1
    f = conv(b1, f"{prefix}.ffn.0", 1, 0, True)
    f = conv(f, f"{prefix}.ffn.1", 1, 0, False)
    return b1 + f

def c2psa(x, i, dump=None):
    cv1 = conv(x, f"model.{i}.cv1", 1, 0, True)
    a, b = cv1.chunk(2, dim=1)
    b = psablock(b, f"model.{i}.m.0", dump)
    cat = torch.cat([a, b], dim=1)
    return conv(cat, f"model.{i}.cv2", 1, 0, True)

@torch.no_grad()
def run():
    caps = {}
    x = torch.from_numpy(ORACLE["input"].astype(np.float32))
    x = conv(x, "model.0", 2, 1, True)
    x = conv(x, "model.1", 2, 1, True)
    x = c3k2(x, 2, True)
    x = conv(x, "model.3", 2, 1, True)
    x = c3k2(x, 4, True); caps["block4"] = x
    x = conv(x, "model.5", 2, 1, True); caps["block5"] = x
    x = c3k2(x, 6, True); caps["block6"] = x
    b6 = x
    x = conv(x, "model.7", 2, 1, True); caps["block7"] = x
    x = c3k2(x, 8, True); caps["block8"] = x
    sd = {}
    x = sppf(x, 9, sd); caps["block9"] = x
    cd = {}
    x = c2psa(x, 10, cd); caps["block10"] = x
    for kk, vv in {**sd, **cd}.items():
        caps[kk] = vv
    b10 = x
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = torch.cat([x, b6], dim=1)
    x = c3k2(x, 13, True)
    b13 = x
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = torch.cat([x, caps["block4"]], dim=1)
    p3 = c3k2(x, 16, True); caps["block16"] = p3
    x = conv(p3, "model.17", 2, 1, True)
    x = torch.cat([x, b13], dim=1)
    p4 = c3k2(x, 19, True); caps["block19"] = p4
    x = conv(p4, "model.20", 2, 1, True)
    x = torch.cat([x, b10], dim=1)
    p5 = c3k2(x, 22, True); caps["block22"] = p5
    return caps

caps = run()

print("=== validate torch reference vs existing oracle (refs.safetensors) ===")
for name in ["block4", "block10", "block16", "block19", "block22"]:
    got = caps[name].numpy()
    want = ORACLE[name]
    d = np.abs(got - want).max()
    print(f"  {name:8s} max|Δ| = {d:.3e}   (oracle range [{want.min():.2f},{want.max():.2f}])")

# Export the extended oracle for the Rust isolation tests.
out = {}
for name in ["block4", "block5", "block6", "block7", "block8", "block9", "block10",
             "sppf_cv1", "sppf_p1", "sppf_p2", "sppf_p3",
             "attn_xattn", "attn_pe", "psa_after_attn"]:
    out[name] = caps[name].numpy().astype(np.float32)
out["input"] = ORACLE["input"].astype(np.float32)
stn.save_file(out, f"{CACHE}/refs_ext.safetensors")
print(f"\nwrote refs_ext.safetensors with {len(out)} tensors")
for k, v in out.items():
    print(f"  {k}: {v.shape}")
