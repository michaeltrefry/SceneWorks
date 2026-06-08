# sc-3635 — Rust SAM2 person segmenter via `ort` (spike) — FINDINGS

**Decision: GO** — via `ort` (onnxruntime), **CPU execution provider** for the Hiera
encoder. Quality matches the Python SAM2 baseline; op coverage is complete on CPU EP;
latency is acceptable for the sampled-frame person-track workload. **CoreML EP is a
net negative for this model and should NOT be used for the encoder** (see §3).

Date: 2026-06-08 · Hardware: Apple M-series (arm64), macOS 25.5 · onnxruntime 1.26.0
(Python) / `ort` 2.0.0-rc.12 (Rust) · model: `onnx-community/sam2.1-hiera-large-ONNX`.

---

## 1. What was validated

The production `_Sam2Segmenter` (`apps/worker/scene_worker/person_adapters.py`):
box prompt → single best mask (`multimask_output=False`, argmax of IoU scores) →
binary `L` mask. Reproduced through three independent legs on real person photos
(`zidane.jpg` 1280×720 two people; `bus.jpg` 810×1080 street scene):

1. **PyTorch baseline** — `transformers` `Sam2Model` on `facebook/sam2.1-hiera-large`
   (MPS). Same official weights / same predictor contract as the worker's
   `sam2_hiera_large.pt` (SAM2.1-large is the same architecture, an incremental
   quality bump over the v1 checkpoint). This is the quality source-of-truth.
2. **ONNX via onnxruntime** (Python) — the community two-graph image-predictor split
   (`vision_encoder.onnx` + `prompt_encoder_mask_decoder.onnx`), CPU and CoreML EPs.
3. **ONNX via `ort`** (Rust) — the same two graphs through the exact `ort`+CoreML
   scaffold already shipping in `pose_jobs.rs` / `upscale_jobs.rs`.

Box-prompt convention for the graph: dedicated `input_boxes` [B,n,4] in 1024-input
space (orig px × 1024/W, ×1024/H — SAM2 stretches to a square 1024, no letterbox),
plus one dummy padding point (`input_labels = -10`, ignored by the prompt encoder)
to satisfy the graph's required point inputs.

## 2. Quality — GO

IoU of the binary person mask vs the references:

| image | ort-CPU vs PyTorch | ort-CoreML vs PyTorch | CoreML vs CPU (EP equivalence) |
|-------|--------------------|------------------------|--------------------------------|
| zidane | **0.9897** | 0.9898 | **0.9998** |
| bus    | 0.9322¹ | 0.9320 | **0.9992** |

Rust `ort` leg vs the Python onnxruntime CPU mask: **IoU 0.9986 (CPU)**, **0.9983
(CoreML)** on zidane. IoU scores reported by the decoder were identical across all
legs (zidane `[_, 0.977, 0.979]`; PyTorch `[_, 0.975, 0.979]`).

Visual overlays (`/tmp/sc3635/*/overlay_*.png`) confirm tight, correct person
segmentation — e.g. the boxed person in `bus.jpg` is cleanly isolated even though the
prompt box loosely included part of the bus.

¹ The lower bus number is boundary precision between two *good* masks on a
partially-occluded subject with a loose box (decoder score ~0.85), not a failure —
the EP-equivalence number (0.9992) shows the ONNX path itself is faithful; the gap is
ONNX-export-vs-PyTorch numerics on a harder case. Visually clean.

## 3. CoreML EP — works, but a net negative (use CPU EP)

The Hiera **vision transformer fragments badly under CoreML**: only ~42% of encoder
nodes are CoreML-eligible and they split into **50–437 partitions**, so the CPU↔CoreML
hand-off overhead swamps any acceleration. Measured encoder latency (steady-state):

| model | precision | EP | encoder ms | IoU vs large-CPU |
|-------|-----------|-----|-----------:|-----------------:|
| **large** | fp32 | **CPU** | **1517** | 1.0000 |
| large | fp32 | CoreML | 4243 | 0.9998 |
| large | fp16 | CoreML | 1691 | 0.9999 |
| large | fp16 | CPU | 1737 | 0.9999 |
| base-plus | fp32 | CPU | 716 | 0.9774 |
| base-plus | fp16 | CoreML | 762 | 0.9774 |
| tiny | fp32 | CPU | 357 | 0.9678 |
| tiny | fp16 | CoreML | 423 | 0.9677 |

CoreML never beats CPU here — even fp16/ANE (1691ms) trails CPU fp32 (1517ms); fp32
CoreML is 2.8× slower. The Rust leg shows the same shape (CPU ~1.9s vs CoreML ~9.6s
incl. first-run graph compile). **Recommendation: run the SAM2 encoder on the CPU EP**
— this is exactly the "fall back to CPU EP where CoreML doesn't help" pattern the story
cites for `pose_jobs.rs`; for SAM2 it's the primary path, not a fallback. The decoder
is tiny (~10ms) and CoreML-hostile (ScatterND/OneHot/Range/Where) → always CPU.

Note: this is *different* from the Slice-1 YOLO finding (sc-3633), where the CoreML EP
**hangs** on the YOLO11 export and Michael chose to pivot that detector to MLX. SAM2
CoreML does not hang; it merely underperforms CPU. The viable `ort` path for SAM2 is
CPU-EP, which still fully satisfies epic 3482 (`ort` is Rust; no Python). See §6 for
the ort-vs-MLX-coherence decision this raises.

## 4. Latency budget

Per-frame (encoder + ~10ms decoder), CPU EP: **large ≈ 1.5–1.9s**, **base-plus ≈ 0.73s**,
**tiny ≈ 0.37s**. Person-track samples few frames (`PERSON_TRACK_SAMPLE_RATE_FPS = 2.0`,
min 3 / max 24), so worst-case large ≈ 24 × 1.6s ≈ 38s for a clip — acceptable for an
offline job. Model size is a clean speed/quality dial; default **large** to match the
Python baseline's quality, expose base-plus as the speed option.

## 5. Provisioning notes for the implementation

- Graphs: `onnx-community/sam2.1-hiera-large-ONNX` → `onnx/vision_encoder.onnx`
  (+`.onnx_data`, ~888MB fp32) and `onnx/prompt_encoder_mask_decoder.onnx`
  (+`.onnx_data`, ~21MB). Download-on-first-use + env pin, mirroring `pose_jobs.rs`
  weight provisioning; host a mirror under the SceneWorks HF org (cf. the
  `SceneWorks/real-esrgan-onnx` pattern from sc-3489).
- **External-data caveat:** the CoreML EP cannot resolve `.onnx_data` external weights
  when it compiles sub-graphs (`model_path must not be empty`); the fix is to inline
  to a single-file `.onnx` (`onnx.save_model(..., save_as_external_data=False)`).
  Since we're recommending **CPU EP**, external-data graphs load fine as-is — inlining
  is only required if CoreML is ever enabled. (onnx also won't follow the HF-cache
  symlink during inline; stage real copies first.)
- fp16 weights are numerically equivalent (IoU 0.9999) and ~half the size; safe to
  ship if download size matters (no CPU speed benefit).

## 6. Open decision surfaced for Michael (not buried)

Slice 1 (sc-3633, YOLO detector) is **pivoting `ort`→MLX** (CoreML hang). This spike
shows SAM2 is **GO on `ort` CPU-EP today** (proven, reuses shipped scaffolding, meets
quality + latency) and that an MLX SAM2 would be a *large net-new engine port* (no
`mlx-gen` SAM2 exists — full Hiera encoder + mask decoder). The spike's data does not
justify that port: ort-CPU already meets the bar. The only argument for MLX-SAM2 is
*stack coherence* with the MLX YOLO. **Recommendation: implement SAM2 on `ort` CPU-EP
now** and treat an MLX SAM2 as a separate, later epic if all-MLX coherence is desired.
Flagging so the ort-vs-MLX call is yours, given the Slice-1 pivot.

## 7. Implementation status / dependency

The implementation half ("Rust segmenter … wired into Slice-2 track assembly, setting
`maskState` active/generated/degraded like Python `segment_track`, box-mask fallback
for degraded") **depends on Slice 2 (sc-3634), which is in Backlog (unstarted)** — the
Rust worker's `run_person_track` is still a procedural placeholder emitting
`maskState:"deferred"`. The segmenter's core (load + box→mask) is proven by this
spike's Rust crate; wiring + end-to-end validation is gated on Slice 2 producing real
track frames. Sequenced accordingly (real dependency, not a deferral of doable work).

## Artifacts

- `scripts/spikes/sc3635_reference.py` — PyTorch baseline + onnxruntime CPU/CoreML
  legs, op-placement profiling, dumps identical inputs for the Rust leg.
- `scripts/spikes/sc3635_variants.py` — encoder EP/precision/model-size sweep (§3).
- `scripts/spikes/sc3635_ort_sam2/` — standalone Rust `ort` SAM2 segmenter spike.
- `/tmp/sc3635/{zidane,bus}/` — masks, overlays, `summary.json`, `variants.json`
  (regenerate via the scripts; not committed).
