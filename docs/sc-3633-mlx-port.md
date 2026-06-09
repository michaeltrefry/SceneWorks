# sc-3633 — YOLO11 person detector: MLX port plan (epic 3482 / sc-3488)

**Decision (Michael, 2026-06-08):** the detector inference backend is **MLX (mlx-rs)**, not
`ort`+CoreML. The whole Mac stack is MLX; and the CoreML EP *hangs indefinitely* in
`commit_from_file` on the Ultralytics YOLO11 ONNX export (clean single-process repro,
0% CPU at 180s, never errors → no fallback). The `ort` path is abandoned for this model.

## What is already done + verified (committed on this branch)

- `crates/sceneworks-worker/src/person_jobs.rs`: letterbox → decode → NMS → f64 normalize
  (Python `run_person_detect` shape) + `media_jobs.rs` preview/real wiring. **Backend-agnostic.**
- **Real-weights parity PASS:** the exact Rust letterbox+decode+NMS pipeline, replicated in
  Python on the real `yolo11m.onnx`, reproduces `ultralytics.predict`'s 4 person boxes on
  `bus.jpg` to ≤0.1px / 0.001 conf. So the decode that runs *on top of* the model output is proven.
- The committed `Detector` (ort `Session`) is the only piece that gets replaced.

## The MLX port: produce the same `(1,84,8400)` tensor, then existing decode runs unchanged

### Weights (already converted + staged)
`~/Library/Application Support/SceneWorks/cache/person-detect/yolo11m_fused_mlx.safetensors`
- 225 tensors, **conv+BN fused** (via ultralytics `model.fuse()`), so the forward needs no BN.
- Conv weights are **MLX layout `(out, kH, kW, in)`** (already transposed from torch `(o,i,kh,kw)`),
  so load raw — do NOT run them through the engine's `conv2d_weight` (that would transpose again).
- Keys mirror the fused torch state_dict (`model.<i>....conv.weight`/`.conv.bias`; bare detect
  convs `model.23.cv2.*.2.{weight,bias}`, `model.23.cv3.*.2.*`, `model.23.dfl.conv.weight`).
- Reproduce: `YOLO("yolo11m.pt").model.fuse()`, transpose 4-D weights `(0,2,3,1)`, save f32 safetensors.

### Parity oracle (already captured)
`…/cache/person-detect/refs.safetensors`: `input` (1,3,640,640, my exact letterbox of bus.jpg),
plus block outputs `block4/10/16/19/22` and `final` (1,84,8400). Build the forward block-by-block
and assert max-abs-diff vs these (NCHW; MLX runs NHWC → transpose for compare).

### Dataflow (`…/cache/person-detect/dataflow.json`) — 24 blocks
backbone: 0 Conv, 1 Conv, 2 C3k2, 3 Conv, 4 C3k2, 5 Conv, 6 C3k2, 7 Conv, 8 C3k2, 9 SPPF, 10 C2PSA;
neck: 11 Upsample, 12 Concat[-1,6], 13 C3k2, 14 Upsample, 15 Concat[-1,4], 16 C3k2, 17 Conv,
18 Concat[-1,13], 19 C3k2, 20 Conv, 21 Concat[-1,10], 22 C3k2; head: 23 Detect[16,19,22].
Save-list outputs needed later for Concat: blocks 4, 6, 10, 13. (NHWC concat is on the channel axis = last.)

### Module → engine-primitive mapping
- **Conv** (ultralytics) = `mlx_gen::nn::conv2d(x, w, Some(b), stride, pad)` then `silu`. (k3s2p1 downsamples; k1s1p0 pointwise.)
- **Bottleneck** = Conv(k3) → Conv(k3), optional residual add (`mlx_rs::ops::add`).
- **C3k** = CSP: split cv1 output in half (channel), n× Bottleneck on one half, concat, cv2.
- **C3k2** = C2f-style: cv1 → split → n× (C3k or Bottleneck) chained, concat all, cv2. (`c3k` flag per block; m-scale: blocks 2/4 use Bottleneck, 6/8/13/16/19/22 use C3k — confirm per `.m[*]` type at convert time and bake into config.)
- **SPPF** = cv1 → x, p1=maxpool5(x), p2=maxpool5(p1), p3=maxpool5(p2) → concat[x,p1,p2,p3] → cv2.
  maxpool 5×5 s1 p2: no pooling op in mlx_rs → `pad` by 2 with −inf then 24× `maximum` over shifts (or implement a small helper).
- **C2PSA** = cv1 → split → PSABlock(attn + FFN) on one half → concat → cv2.
  - Attention: qkv = conv1×1; split q,k,v; MHA with `matmul` + `softmax_axis(-1)`; proj conv. (head dim from `Attention` cfg.)
- **Upsample** = `mlx_gen::nn::upsample_nearest(x, 2)`.
- **Concat** = `mlx_rs::ops::concatenate_axis(&[...], -1)` (channel-last).
- **Detect head (block 23)**: three branches (cv2 box-reg → 64ch, cv3 cls → 80ch). Then:
  - DFL: box-reg reshape `(b, 4, 16, A)` → `softmax_axis(2)` → matmul with `[0..15]` → `(b,4,A)` distances.
  - dist2bbox with **precomputed anchor points + strides** (grids at strides 8/16/32 for 640 → 80²+40²+20²=8400 anchors; anchor = cell center +0.5; ltrb → xywh in input px). Compute these as Rust constants.
  - cls = `sigmoid`. Concat `[xywh(4), cls(80)]` on axis1 → `(1,84,8400)` == `refs.final`. Existing `decode()` consumes this.

### Where it plugs in
Replace `Detector`/`build_session`/ort imports in `person_jobs.rs` with a `YoloMlx` struct
(load weights once, cached like the ort one), `detect(img)->Vec<Detection>` = preprocess (reuse
existing letterbox, but emit NHWC f32 Array) → forward → existing `decode`+`nms`. Keep the
`spawn_blocking` + process-wide cache. Drop the `ort`/`zip` use for person-detect (ort stays for pose_jobs).

### Parity test
`#[ignore]` test: load `refs.safetensors`, run forward, assert per-block + final max-abs-diff < ~1e-3
(fused-conv fp32), then run `decode`+`nms` and assert the 4 bus.jpg boxes vs `ref_people.json` (≤2px).

## Open API confirmations for the next session
- The engine `Weights` loader: how to construct from a `.safetensors` path + `.require(name)->Array`
  (used across mlx-gen; e.g. `mlx-gen-wan/tests/*` `fn load(name)->Weights`). Confirm the public path.
- `Array` channel-split (`split`/slicing) + `transpose_axes` for the NCHW↔NHWC compares.

## DONE — native mlx-rs YOLO11m forward implemented + parity-verified (2026-06-09)

The full forward is in `crates/sceneworks-worker/src/person_jobs.rs` (`Yolo`/`YoloForward`),
built from `mlx_gen::nn::{conv2d,silu,upsample_nearest}` + `mlx_rs::ops`. The `ort` `Detector`,
`build_session`, and the `ort`/`zip` imports are gone from the person-detect path (`ort` stays
for `pose_jobs`). `mlx-rs` (pmetal-mlx-rs, same rev mlx-gen pins) is now a first-class macOS
dependency. `media_jobs.rs` reports `backend:"mlx"`, adapter `yolo11_mlx`, device `mlx`.

### Architecture, straight from the fused state-dict (authoritative source of truth)
- **Every** C3k2 block (2/4/6/8/13/16/19/22) is `c3k=True, n=1` with an inner `C3k(n=2)` — the
  `model.<i>.m.0.cv3` + `m.0.m.{0,1}` keys prove it. So all eight are structurally identical,
  differing only in channel dims (read from the weights). The standard-YAML "c3k=False for early/
  neck blocks" does NOT hold for this checkpoint — trust the weights, not the yaml.
- Bottleneck residual is on everywhere: C3k2 defaults `shortcut=True` and the yaml never overrides
  it; all inner bottlenecks have `c1==c2`. (Confirmed: block4 parity is near-bit-exact.)
- C2PSA (block 10): `num_heads=4, key_dim=32, head_dim=64`, scale `32^-0.5`; PE is a depthwise 3×3
  on `v` reshaped to a feature map; ffn.1 has no activation.
- Detect: DFL bin indices are read straight from `model.23.dfl.conv.weight` ([0..15]); anchors are
  cell-center (+0.5) row-major over 80²@8 / 40²@16 / 20²@32; output assembled `[xywh, sigmoid(cls)]`
  → transposed to `(1,84,8400)` channel-major for the existing `decode()`.

### Two implementation gotchas
- **`mlx_rs::ops::conv2d` only supports `groups=1`** → depthwise convs (C2PSA `pe`, Detect `cv3`
  DWConv, all `[C,3,3,1]`) are a manual 9-tap shift/multiply/accumulate (pad-by-1, per-channel tap
  broadcast over a `try_index` range slice). SPPF 5×5 max-pool uses the same shift trick (−inf pad,
  25-tap elementwise `maximum`).
- **`Array::as_slice` returns the PHYSICAL buffer, not the logical order.** The head ends in a
  `transpose_axes`, so the `(1,84,8400)` output is a non-contiguous view; calling `as_slice` on it
  directly hands back the pre-transpose `(1,8400,84)` data → `decode` reads scrambled rows (the
  symptom was 327 post-NMS boxes). Fix: `reshape(&[-1])` first to force a logical-order copy. (The
  oracle test's `flat()` already did this, which is why per-block parity passed while the e2e path
  failed — a useful tell.)

### Parity results (`cargo test -p sceneworks-worker person_jobs -- --include-ignored --test-threads=1`)
Run MLX tests single-threaded — concurrent MLX forwards across test threads corrupt results.
- Per-block vs `refs.safetensors`: block4 **1.7e-5** (near-bit-exact → backbone correct); block10
  6.8e-3, block16 5.4e-3, block19 1.3e-2, block22 7.7e-3. **NOTE (sc-3734): the "benign fp32 drift"
  label below was WRONG.** The divergence was MLX's Metal `matmul` reduced-precision simdgroup path
  (≈1e-3 relative — genuine fp32 drift here is ~1e-5) entering only through the C2PSA attention's two
  SDPA matmuls. Fixed by running those matmuls on the CPU stream (`attention()`); every block is now
  ≤1.5e-4 and the oracle thresholds are re-tightened to 1e-3. See `docs/sc-3734/findings.md`.
- Final head: class rows **1.98e-3** (now 1.9e-5 post-fix), box rows **<1px**.
- End-to-end (letterbox→forward→decode→NMS) reproduces `ultralytics.predict`'s **4 people on
  people.jpg to ≤2px / 0.02 conf** (0.917/0.907/0.906/0.778). Decode/NMS already verified in #485.

Weight provisioning (download-on-first-use, HF host) is still slice 4 / sc-3636 — the resolver reads
the staged `yolo11m_fused_mlx.safetensors` from the app cache / model dir for now.
