# Epic 3494 — Candle SDXL Windows feasibility spike (GO, with hardening)

**Decision: GO.** Hugging Face **Candle** is a viable Rust-native Windows/CUDA SDXL
inference path — proven end-to-end on RTX PRO 6000 Blackwell (sm_120). It builds,
runs sm_120, and produces correct 1024² SDXL images through a worker-shaped contract.
But the naive prototype is **not production-competitive as-is** (~3.6× slower, ~2.2×
VRAM vs the Python-Torch path, plus a seed-reproducibility defect — all with known
fixes), so continue into a hardening + integration slice. **No Python-replacement
claim**: the Candle lane would be SDXL/RealVisXL **txt2img only**, parallel to the
existing macOS mlx-gen lane; the Python worker stays primary for everything else.

Spike stories sc-3495–3499 all Done. Follow-up implementation work: epic **3672**
(stories sc-3673–3678).

## Per-story findings
- [`sc-3495-cuda-viability.md`](sc-3495-cuda-viability.md) — CUDA build/runtime on
  Windows/Blackwell + the MSVC-toolset gotcha + distribution/packaging analysis.
- [`sc-3496-contract.md`](sc-3496-contract.md) — SceneWorks SDXL contract → Candle
  provider mapping (REQUIRED / DEFERRED / NOT NEEDED).
- [`sc-3497-prototype.md`](sc-3497-prototype.md) — gated SDXL txt2img prototype
  through a worker-shaped asset/reporting contract.
- [`sc-3498-validation.md`](sc-3498-validation.md) — head-to-head vs Python-Torch
  (perf, VRAM, reproducibility) + go/no-go.

## Headline numbers (Blackwell sm_120, 1024², 30 steps, guidance 7.0, seed 42)
| Metric | Python-Torch (diffusers, bf16) | Candle prototype (f16) |
|---|---|---|
| mean step | 0.089 s (~11 it/s) | ~0.32 s (~3.1 it/s) — ~3.6× slower |
| peak VRAM | 8.95 GiB | ~19.7 GiB — ~2.2× |
| quality | high | high (typical seeds) |
| seed reproducibility | portable (torch.Generator) | env-fragile, occasional collapse |

## Critical gotchas (carry into implementation)
1. **MSVC toolset:** CUDA 12.9's nvcc rejects VS 2026's MSVC 14.51 — build with the
   VS 2022 **v143** toolset (build-time only).
2. **VAE scale 0.13025** for SDXL (candle's example hardcodes the SD1.5 value 0.18215).
   fp16-VAE-fix required in f16.
3. **Reproducibility:** don't rely on candle CUDA `set_seed`; seed initial latents on
   CPU with a fixed algorithm (like diffusers) + use a non-ancestral scheduler.
4. **Distribution:** one CI multi-arch fatbin + bundled CUDA redist DLLs (cudarc
   dynamic-linking) — same model torch uses, not per-GPU builds.

## Evidence images
- `images/candle-sdxl-seed42.png` — Candle SDXL, seed 42 (good).
- `images/python-sdxl-seed42.png` — Python-Torch SDXL, seed 42 (good).
- `images/candle-seed42-repro-collapse.png` — same seed 42, different launch env →
  reproducible collapse (the reproducibility defect).

Prototype/harness code lives out-of-repo at `D:\sceneworks-candle-spike\`
(`candle-smoke` crate, `src/bin/candle_sdxl.rs`, `validate_sdxl.py`, build scripts).
