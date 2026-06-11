# sc-4387 — DWPose backend: native-MLX port vs keep `ort`+CoreML (R2)

**Recommendation: do NOT port DWPose to MLX in isolation.** On the evidence, DWPose
is the *success* case for the CoreML EP — not a failure case like YOLO11 (hung) or
SAM2 (fragmented). Keep `ort`+CoreML for DWPose now; treat an MLX port as a deliberate
*consistency* play to be decided **jointly with the Real-ESRGAN upscaler (sc-3489)**,
since porting DWPose alone removes zero dependencies and buys no functional win.

Confidence: MODERATE (~65%). The functional/data case for keeping is strong; the case
*for* porting is purely strategic (single-backend coherence + retiring the CoreML-EP /
bundled-dylib maintenance tax) and only pays off if the upscaler moves too. The
port-vs-keep call is Michael's — this spike supplies the evidence, not the verdict.

Date: 2026-06-11 · Hardware: Apple M-series (arm64), macOS 25.5 · onnxruntime 1.26.0 ·
models: production pins `yolox_m_8xb8-300e_humanart` + `rtmw-dw-x-l_simcc-cocktail14`.

---

## 1. "Does CoreML even belong here?" — YES (measured)

CoreML-EP node partitioning on the two production exports (verbose `GetCapability` +
`VerifyEachNodeIsAssignedToAnEp`, onnxruntime 1.26):

| export       | CoreML partitions | CPU-only nodes | why the CPU nodes |
|--------------|-------------------|----------------|-------------------|
| rtmw-dw-x-l  | **2**             | 48             | GAU head: `MatMul`/`ReduceSum`/`Sqrt`/`Div` (ScaleNorm) + SimCC projections + `DepthToSpace` |
| yolox_m      | 8                 | 36             | embedded-NMS tail (`NonMaxSuppression`/`TopK`/`Where`) + rank>5 reshape tensors from the FPN `Resize` |

This is the **opposite** of the SAM2 Hiera result (50–437 partitions, sc-3635). The
heavy rtmw conv backbone runs as **2 clean CoreML islands**; only the light GAU+SimCC
head (48 small nodes) falls back to CPU. yolox fragments more (8), but its conv
backbone is still on CoreML and the fragmentation is the NMS tail we wouldn't port
anyway. **CoreML is genuinely well-suited to these CNNs.**

## 2. Latency (this machine, steady-state, n=20)

| net          | onnxruntime CPU | CoreML | CoreML speedup |
|--------------|-----------------|--------|----------------|
| rtmw-dw-x-l  | 37.5 ms         | **15.6 ms** | 2.4× |
| yolox_m      | 70.8 ms         | ~17–25 ms (sc-3487)¹ | ~3–4× |

End-to-end per image (det+pose) ≈ **35–55 ms on CoreML**, confirming sc-3487. CoreML
warmup here was ~32 ms (compiled model already cached this session; first-ever compile
is still the ~4.4 s sc-3487 reported).

¹ The random-noise latency harness can't time yolox CoreML: with no real person in the
frame the embedded NMS emits 0 boxes → a zero-element dynamic tensor the CoreML EP
rejects (`coreml_execution_provider.cc:198`). Benign artifact of synthetic input; also
a live demonstration that the NMS tail is CoreML-hostile. yolox CoreML latency is taken
from sc-3487 (real images).

## 3. Native-MLX feasibility — HIGH (~90%)

Op inventory from the real ONNX graphs:

- **yolox_m** (25.3M params, 415 nodes): `Conv 107, Mul 102, Sigmoid 100` (CSPDarknet +
  SiLU), `MaxPool 3` (SPPF), `Resize 2` (FPN), `Concat 20`. This is **op-for-op the
  YOLO11 set already implemented** in `person_jobs.rs`. The embedded-NMS tail wouldn't
  be ported — like YOLO11, convert the raw weights and do greedy NMS in pure Rust
  (already written). Essentially a solved problem.
- **rtmw-dw-x-l** (57.3M params, 464 nodes): CSPNeXt SiLU backbone (`Conv 122,
  Sigmoid 116, MaxPool 3, Resize 2`) + `GlobalAveragePool 4`/`HardSigmoid 4`/`Relu 4`
  (4 SE channel-attention blocks) + `ReduceSum 3`/`Sqrt 3`/`Div 4` (GAU ScaleNorm) +
  `DepthToSpace 1` (head pixel-shuffle) + **`MatMul 8`** (GAU attention + 2 SimCC
  projections). SimCC argmax decode is **already in pure Rust** (`pose_jobs.rs:218`).

There are **no exotic ops**. The only genuinely net-new module is the GAU attention
block (assemblable from existing mlx-gen attention); SE-attention, ScaleNorm and
DepthToSpace are a few lines each.

**Depthwise convs use native fused grouped conv — NOT the slow shift-trick.** mlx-rs
`ops::conv2d` takes a `groups` arg and depthwise works at the worker's exact pin
(`1db3cd0`): the SAM2 memory encoder runs `groups=channels` 7×7 depthwise convs with a
passing channel-isolation test (`mlx-gen-sam2/src/memory.rs`). CSPNeXt's
depthwise-separable backbone maps cleanly onto that. (YOLO11's 9-tap shift-trick was a
choice — it used the groups=1 `mlx_gen::nn` wrapper — not a limitation.)

## 4. Parity risk (reduced-precision Metal matmul) — LOW, cheap to handle

Per the known MLX-Metal matmul reduced-precision issue (~1e-3, affects matmuls not
convs), the at-risk ops are the **8 MatMuls** — and the parity-critical ones are just
the 2 final SimCC projections feeding the bin argmax (split 2.0 → 0.5 px/bin). A
targeted `matmul_device(cpu)` detour on those 8 holds the sc-3487 sub-pixel bar without
touching the conv bulk that dominates latency. This does **not** materially fight the
latency case.

## 5. Effort + MLX latency outlook

- **Effort ≈ 2–2.5× the YOLO11 port**: two nets (yolox ≈0.8× YOLO11; rtmw ≈1.3–1.5×) +
  conv+BN-fuse weight conversion for both + HF hosting (SceneWorks org, cf.
  `yolo11m-person-detect-mlx`) + per-block parity oracles. Consistent with the story's
  "SimCC head is heavier than YOLO11's decode."
- **MLX latency estimate (NOT measured): ~20–45 ms rtmw** = ~1.3–3× CoreML's 15.6 ms.
  MLX runs the Metal GPU; CoreML gets the ANE + a compiled fused graph. With native
  grouped conv the catastrophic-slow case is ruled out, but MLX is **very unlikely to
  beat CoreML** — best case it matches. So MLX latency offers **no functional win**, only
  a possible small regression. A full MLX forward (= most of the implementation) was
  *not* built for this spike because it cannot flip the recommendation; the exact number
  should be confirmed during the implementation story if porting is greenlit.

## 6. The decision the data actually supports

1. **Dropping DWPose's `ort` does not remove `ort`/CoreML/the bundled
   `libonnxruntime.dylib`** — the Real-ESRGAN upscaler (`upscale_jobs.rs`, sc-3489) uses
   the same stack and the same dylib. Porting DWPose alone removes call sites, **zero
   crates**. The dependency-surface payoff is conditional on sc-3489.
2. **Epic 3482's bar is zero-*Python*, and `ort` is Rust** — DWPose on `ort`+CoreML
   already satisfies the north star. "Consistency" here means MLX-vs-CoreML backend
   uniformity, an architecture preference, not a Python-eradication requirement.
3. **Precedent (sc-3635 SAM2): keep `ort` when it works.** SAM2 stayed on `ort` (CPU-EP)
   rather than forcing an MLX port the data didn't justify — and SAM2's CoreML was a net
   *negative*. DWPose's CoreML is a net *positive* (2 partitions, 2.4× CPU), so the case
   to keep is even stronger than SAM2's.
4. **YOLO11 went MLX because CoreML was impossible** (hang), not because MLX was
   preferred. That precedent does not transfer to DWPose, where CoreML works well.

### Ranked options
1. **(Preferred, ~65%) Keep `ort`+CoreML for DWPose.** Proven, fast, zero-Python,
   matches the SAM2 precedent. Revisit MLX only as a joint decision with sc-3489.
2. **(If single-backend consistency is the priority) Port DWPose + upscaler together**
   as one "retire `ort`+CoreML from the Mac worker" epic — the only framing where the
   dependency/consistency payoff is real, accepting ~2–2.5× YOLO11 effort and a possible
   small latency regression as the price.
3. **(Argue against) Port DWPose alone for "consistency"** — worst of both: full effort,
   no dependency removal, possible regression, against the SAM2 precedent.

R3 (placement) follows R2: if it stays `ort`, it stays worker-local; if it's ported, it
becomes an `mlx-gen-dwpose` utility crate (sam2/face pattern).

## Reproduce

```
~/.dwpose-spike/venv/bin/python scripts/spikes/sc4387_dwpose_diag.py
# op histogram + CoreML partition counts + CPU/CoreML latency (rtmw on random input;
# yolox CoreML needs a real image — see §2 note).
```
