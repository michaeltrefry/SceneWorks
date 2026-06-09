#!/usr/bin/env python
"""sc-3734 — confirm the C2PSA x_attn loss is INSIDE MLX's Metal matmul, not our algorithm.

Consumes the one-off dump written by the Rust `yolo11_mlx_per_block_isolation` test when run
with SC3734_DUMP=1 (MLX's OWN post-softmax `attn`, `vh`, and the GPU `x_attn`), then recomputes
attn·v in fp64 from those exact MLX inputs. If MLX's GPU x_attn differs from the fp64 recompute
of MLX's own attn/vh, the precision loss is the matmul kernel itself.

  SC3734_DUMP=1 cargo test -p sceneworks-worker \
    person_jobs::tests::yolo11_mlx_per_block_isolation -- --ignored --test-threads=1
  ~/mlx-flux-venv/bin/python docs/sc-3734/mlx_matmul_check.py

Expected (M-series, mlx-rs rev e59ffd88):
  MLX attn vs fp64(MLX q·k→softmax) ... actually tested via the av step below
  MLX x_attn vs fp64(MLX_attn @ MLX_vh) = ~4.8e-3   <-- loss is inside MLX's av-matmul
A faithful fp32 av-matmul would be ~1e-6 (see attn_precision.py).
"""
import numpy as np

N = 400
def load(name, shape):
    return np.fromfile(f"/tmp/sc3734_{name}.f32", dtype="<f4").reshape(shape)

attn = load("attn", (4, N, N))        # MLX post-softmax weights
vh = load("vh", (4, N, 64))           # MLX values
xattn_mlx = load("xattn_chw", (1, 256, 20, 20)).reshape(4, 64, 400)  # (head,e,n)

print(f"attn row-sum range (sanity, ~1): {attn.sum(-1).min():.6f} .. {attn.sum(-1).max():.6f}")
ref64 = np.einsum("hij,hje->hei", attn.astype(np.float64), vh.astype(np.float64))
d = np.abs(xattn_mlx.astype(np.float64) - ref64).max()
print(f"MLX x_attn vs fp64(MLX_attn @ MLX_vh) max|Δ| = {d:.3e}")
print("  -> >1e-3 means the loss is inside MLX's Metal matmul (reduced-precision simdgroup),")
print("     since the inputs are MLX's own and the recompute is exact fp64.")
