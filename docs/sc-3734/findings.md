# sc-3734 — YOLO11-MLX per-block divergence: root cause + fix

**Verdict (high confidence):** the ~1e-2 block10/19 divergence labelled "benign fp32
backend drift" in sc-3633 was **NOT fp32 drift**. It was **MLX's Metal `matmul` running a
reduced-precision (≈1e-3 relative, tf32/bf16-class simdgroup) accumulation path**, entering
*only* through the two scaled-dot-product matmuls in the C2PSA (block 10) attention. Fixed
by running those two matmuls on the MLX **CPU stream** (true fp32). Every other primitive —
all convs, the SPPF max-pool, and the depthwise positional-encoding — was already faithful.

## How it was isolated (no guessing — measured)

1. **Faithful torch reference** (`torch_ref.py`): the YOLO11m forward reassembled from the
   *same* fused weights, fp32 on CPU. It reproduces the existing oracle (`refs.safetensors`)
   at **every** captured point to ~1e-5 (block4 1.7e-5 … block10 2.0e-5 … final). This both
   (a) confirms the oracle's provenance/dtype is sound (an open question in the story) and
   (b) yields clean ground-truth for the intermediate blocks 5–9 + C2PSA sub-steps, exported
   as `refs_ext.safetensors`.

2. **Per-block isolation** (Rust `yolo11_mlx_per_block_isolation`): feed each block the
   *previous* block's clean ground-truth and run only that block's Rust op — intrinsic error,
   zero accumulation:

   | block | op | intrinsic max\|Δ\| |
   |---|---|---|
   | 5 | Conv | 3.1e-5 |
   | 6 | C3k2 | 1.1e-5 |
   | 7 | Conv | 2.4e-5 |
   | 8 | C3k2 | 9.1e-6 |
   | **9** | **SPPF** | **2.3e-5** — exonerated |
   | **10** | **C2PSA** | **6.8e-3** — the entire source |

3. **Inside C2PSA**: `pe` (depthwise3x3) = 3.6e-6 (exonerated); `x_attn = softmax(q·kᵀ)·v`
   = 7.0e-3 → the whole error is the two attention matmuls + softmax.

4. **bug vs drift** (`attn_precision.py`): a faithful fp32 attention — in *any* value-
   reduction order — matches fp64 to **~6e-6**, and two different fp32 orderings differ by
   **4.3e-6**. Genuine fp32 reduction-order drift here is ~6e-6, so **7e-3 is ~1000× too
   large to be drift.** Recomputing in fp64 from MLX's *own* dumped intermediates:
   - MLX `attn` (post-softmax) vs fp64(MLX q·k → softmax) = **1.26e-3** (qk-matmul is lossy)
   - MLX `x_attn` vs fp64(MLX attn @ MLX vh) = **4.80e-3** (av-matmul is lossy, same inputs)

   Both MLX matmuls lose ~1e-3 relative on identical fp32 inputs — the signature of a
   reduced-precision Metal matmul kernel (between fp16- and bf16-input rounding in
   magnitude; consistent with a tf32-style split-precision simdgroup path), **not** our
   algorithm (which is byte-for-byte the torch reference that matches the oracle).

## The fix

`person_jobs.rs::attention()` — the two SDPA matmuls use `matmul_device(…, StreamOrDevice::cpu())`:

```rust
let cpu = mlx_rs::StreamOrDevice::cpu();
let attn = multiply(&qh.matmul_device(&kh, &cpu)?, &scale)?;
let attn = softmax_axis(&attn, -1, true)?;
let out = attn.matmul_device(&vh, &cpu)?.transpose_axes(&[1, 0, 2])?;
```

Result — full forward vs the oracle, before → after:

| point | before | after | tol |
|---|---|---|---|
| block10 | 6.8e-3 | 2.5e-5 | 1e-3 |
| block16 | 5.4e-3 | 1.9e-5 | 1e-3 |
| block19 | 1.26e-2 | 1.55e-4 | 1e-3 |
| block22 | 7.7e-3 | 3.3e-5 | 1e-3 |
| final cls | 1.98e-3 | 1.9e-5 | 1e-3 |
| final box | 0.91px | 0.55px | 1px |

Oracle thresholds re-tightened from the loosened 2e-2 → 1e-3 (the story's success criterion:
drive every block <1e-3). E2E still reproduces ultralytics' 4 people on people.jpg. The
isolation test is kept as a permanent guard (asserts every block <1e-3 + pins the GPU-vs-CPU
matmul contrast). `x_attn` on the CPU stream is 4.9e-6 vs 7.0e-3 on the raw GPU matmul.

Cost: the C2PSA map is tiny (N=400, 4 heads, once per forward) so the CPU detour is
negligible; e2e timing unchanged (~0.2s).

## Implications beyond this story

This is an **MLX-wide property**, not a YOLO bug: any parity-critical MLX port that uses
`matmul`/attention on Metal inherits ≈1e-3 relative error per matmul. Relevant to:
- **SAM2 (epic 3704)** — segmentation masks are parity-sensitive; but SAM2's attention maps
  are far larger (e.g. 64²=4096 tokens), so a blanket CPU-stream detour there could be a real
  perf hit. The lever (CPU-stream matmul) is the same; whether to pay it is a per-model call.
- Diffusion engines (FLUX/Qwen/SDXL in mlx-gen) tolerate it — sampling is inherently noisy —
  which is why it never surfaced before a bit-exact CNN parity oracle made it visible.

## Repro
```
~/mlx-flux-venv/bin/python docs/sc-3734/torch_ref.py        # writes refs_ext.safetensors
~/mlx-flux-venv/bin/python docs/sc-3734/attn_precision.py    # needs the SC3734_DUMP /tmp dump
cargo test -p sceneworks-worker person_jobs::tests::yolo11_mlx -- --ignored --nocapture --test-threads=1
```
