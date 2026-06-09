#!/usr/bin/env python
"""sc-3734 — characterize the C2PSA attention numerical sensitivity in fp32.

The Rust per-block isolation pinned the entire ~7e-3 block10 divergence to
x_attn = softmax(scale·q·kᵀ)·v (the depthwise PE is 3.6e-6, exonerated). The Rust
algorithm is identical to the torch reference, which matches the oracle to 2e-5 — so the
gap is Metal-fp32 vs CPU-fp32. This script answers: is ~7e-3 consistent with genuine
fp32 reduction-order sensitivity (drift) or does it require a precision loss beyond
reordering (which would smell like a backend defect / our bug)?

It recomputes x_attn from the SAME clean block9 ground truth in:
  - fp64 (truth)
  - fp32 with the torch op order
  - fp32 with several permuted reduction orders (simulating different valid fp32 backends)
and reports the spread, plus the value magnitudes that condition the error.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import safetensors.numpy as stn

CACHE = f"{os.environ['HOME']}/Library/Application Support/SceneWorks/cache/person-detect"
W = {k: v.astype(np.float32) for k, v in stn.load_file(f"{CACHE}/yolo11m_fused_mlx.safetensors").items()}
EXT = stn.load_file(f"{CACHE}/refs_ext.safetensors")

def conv_np(x, prefix, dtype):
    # x: (B,C,H,W); weight fused (o,kh,kw,i)
    w = torch.from_numpy(W[f"{prefix}.conv.weight"].transpose(0, 3, 1, 2).copy()).to(dtype)
    b = torch.from_numpy(W[f"{prefix}.conv.bias"]).to(dtype)
    y = F.conv2d(torch.from_numpy(x).to(dtype), w, b)
    return F.silu(y).numpy()

block9 = EXT["block9"]  # (1,512,20,20) clean torch ground truth, NCHW
ref_xattn = EXT["attn_xattn"].astype(np.float64)  # (1,256,20,20)

# cv1 -> split -> b half (the attention input), in fp64 for a clean baseline.
cv1 = conv_np(block9, "model.10.cv1", torch.float64)  # (1,512,20,20)
b_half = cv1[:, 256:, :, :]  # (1,256,20,20)
B, C, H, Wd = b_half.shape
N = H * Wd
nh, kd, hd = 4, 32, 64

def project(dtype):
    w = torch.from_numpy(W["model.10.m.0.attn.qkv.conv.weight"].transpose(0, 3, 1, 2).copy()).to(dtype)
    bb = torch.from_numpy(W["model.10.m.0.attn.qkv.conv.bias"]).to(dtype)
    qkv = F.conv2d(torch.from_numpy(b_half.astype(np.float64)).to(dtype), w, bb).numpy()
    qkv = qkv.reshape(B, nh, 2 * kd + hd, N)
    q = qkv[:, :, :kd, :]
    k = qkv[:, :, kd:2 * kd, :]
    v = qkv[:, :, 2 * kd:, :]
    return q, k, v

def attn_forward(q, k, v, dtype, perm=None):
    q = q.astype(dtype); k = k.astype(dtype); v = v.astype(dtype)
    scale = np.array(kd ** -0.5, dtype=dtype)
    # attn = (qᵀk)·scale  -> (B,nh,N,N)
    logits = np.einsum("bhdn,bhdm->bhnm", q, k, optimize=True).astype(dtype) * scale
    # stable softmax in the working dtype
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp((logits - m).astype(dtype)).astype(dtype)
    a = (e / e.sum(axis=-1, keepdims=True)).astype(dtype)  # (B,nh,N,N)
    if perm is not None:
        a = a[:, :, :, perm]
        vv = v[:, :, :, perm]
    else:
        vv = v
    # x_attn[b,h,e,i] = sum_j a[b,h,i,j]*v[b,h,e,j]
    x = np.einsum("bhij,bhej->bhei", a, vv, optimize=True).astype(dtype)
    return x.reshape(B, C, H, Wd)

q64, k64, v64 = project(torch.float64)
x64 = attn_forward(q64, k64, v64, np.float64)

q32, k32, v32 = project(torch.float32)
x32 = attn_forward(q32, k32, v32, np.float32)

print("=== value magnitudes (condition the error) ===")
print(f"  |q| max {np.abs(q64).max():.2f}   |k| max {np.abs(k64).max():.2f}   |v| max {np.abs(v64).max():.2f}")
logits = (np.einsum('bhdn,bhdm->bhnm', q64, k64) * (kd ** -0.5))
print(f"  qk logits range [{logits.min():.2f}, {logits.max():.2f}]   (large => peaky softmax)")
print(f"  x_attn range [{x64.min():.2f}, {x64.max():.2f}]")

print("\n=== fp64 reference vs torch-captured oracle x_attn ===")
print(f"  max|Δ| = {np.abs(x64 - ref_xattn).max():.3e}  (validates the numpy reimplementation)")

print("\n=== fp32 (torch op order) vs fp64 truth ===")
print(f"  max|Δ| = {np.abs(x32.astype(np.float64) - x64).max():.3e}")

print("\n=== fp32 under permuted value-reduction orders (simulates different fp32 backends) ===")
rng = np.random.default_rng(0)
spread = []
for t in range(6):
    perm = rng.permutation(N)
    xt = attn_forward(q32, k32, v32, np.float32, perm=perm)
    d = np.abs(xt.astype(np.float64) - x64).max()
    spread.append(d)
    print(f"  perm {t}: max|Δ vs fp64| = {d:.3e}")
print(f"  -> fp32 ordering-induced spread: {min(spread):.3e} .. {max(spread):.3e}")

print("\n=== pairwise max|Δ| between two different fp32 orderings (backend-vs-backend proxy) ===")
xa = attn_forward(q32, k32, v32, np.float32, perm=rng.permutation(N))
xb = attn_forward(q32, k32, v32, np.float32, perm=rng.permutation(N))
print(f"  max|Δ| = {np.abs(xa.astype(np.float64) - xb.astype(np.float64)).max():.3e}")
