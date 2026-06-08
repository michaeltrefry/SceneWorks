# sc-3498 — Candle vs Python-Torch SDXL: validation + go/no-go

Epic 3494. Head-to-head of the Candle SDXL prototype against the Python-Torch
diffusers path that `SdxlDiffusersAdapter` wraps (`StableDiffusionXLPipeline`),
matched prompt/seed/dims/steps/guidance. Same GPU, run sequentially.

## Environment
- GPU: RTX PRO 6000 Blackwell (sm_120), driver 596.36, Windows 11.
- Python ref: torch **2.11.0+cu128**, diffusers 0.37.1, Python 3.12 (3.14 has no
  torch wheels), bf16, EulerDiscreteScheduler (SDXL shipped default).
- Candle: candle-core 0.10.2 git main `3d3d9c41`, CUDA 12.9, MSVC v143 toolset,
  f16, Euler-ancestral scheduler.
- Prompt: "a photo of a rusty robot holding a lit candle, dramatic cinematic
  lighting, highly detailed"; 1024×1024, 30 steps, guidance 7.0, seed 42.

## Head-to-head

| Metric | Python-Torch (diffusers) | Candle prototype | Delta |
|---|---|---|---|
| dtype | bf16 | f16 | — |
| scheduler | EulerDiscrete | Euler-ancestral | parity gap |
| mean step | **0.089 s** (~11.2 it/s) | **~0.32 s** (~3.1 it/s) | **Candle ~3.6× slower** |
| 30-step gen | 3.16 s | ~9.5 s | — |
| model load | 4.52 s | ~3.5 s | ~ |
| total (load+gen) | ~7.7 s | ~13 s | — |
| peak VRAM | **8.95 GiB** | **~19.7 GiB** | **Candle ~2.2×** |
| output dims | 1024² | 1024² | match |
| output quality | high | high (typical seeds) | comparable |
| seed reproducibility | portable (torch.Generator) | env-fragile (see below) | **gap** |
| worker contract | n/a | assetWrites + sidecar + progress | match |

Both produce correct, high-quality SDXL images (robot+candle). Not pixel-identical
(different RNG + scheduler) — expected, never claimed.

## Key finding: reproducibility is environment-fragile (and occasionally collapses)

- **Within one launch environment, Candle is fully deterministic**: seed 99 run
  twice via the same path → byte-identical PNG (md5 match).
- **Across launch environments, the same integer seed maps to different noise**:
  seed 42 via `cargo run` → a good robot (md5 882D…); seed 42 via a `Start-Process`
  launch → a degenerate foliage image (md5 E89A…), reproducibly. Same binary, seed,
  prompt — only the launch environment differs.
- Root cause: the prototype relies on candle's CUDA `device.set_seed()` + `randn`
  for the initial latents, and the seed→noise mapping is **not portable** across
  runtime conditions. diffusers avoids this by seeding a `torch.Generator` with a
  defined algorithm, so a seed reproduces everywhere. Occasionally a bad noise draw
  collapses the sample to garbage.
- **Fix path (known, unimplemented):** generate the initial latent noise with a
  deterministic seeded CPU RNG (fixed algorithm) and move it to GPU — exactly what
  diffusers does — instead of relying on candle CUDA `set_seed`. Use a non-ancestral
  scheduler (DDIM / UniPC, both in candle) to also remove per-step stochastic noise.
  This should make output portably reproducible per seed and eliminate the collapse.

## Other gaps (all with known fixes)
- **Perf (3.6× slower):** prototype is naive — f16, `use_flash_attn=false`, no fused
  attention. Candle has a flash-attn feature; torch's lead is cuDNN/SDPA/flash. A
  fair optimized comparison needs flash-attn + kernel tuning before judging Candle's
  ceiling. The 3.6× is a *prototype* number, not Candle's floor.
- **VRAM (2.2×):** prototype loads both CLIP text encoders in **F32** (per candle's
  reference) and uses no attention slicing / VAE tiling. Loading CLIP in f16 + slicing
  should close most of the gap.
- **Scheduler:** Euler-ancestral (Candle) vs Euler-discrete (SceneWorks). Switching
  to a discrete/deterministic scheduler is part of the reproducibility fix.

## Go / No-Go

**GO, with an optimization + correctness-hardening phase before production.**

Candle SDXL on Windows/Blackwell is *proven viable*: it builds, runs sm_120, and
produces correct high-quality 1024² SDXL images through a worker-shaped contract.
But the naive prototype is **not production-competitive as-is**: ~3.6× slower, ~2.2×
VRAM, and a seed-reproducibility defect. All three have concrete, known fixes
(flash-attn + f16 CLIP + slicing; deterministic CPU-seeded latents + non-ancestral
scheduler). Recommend continuing into a hardening slice, NOT shipping the prototype.
Do not claim Python replacement from this — the perf/VRAM/repro gaps are real today.
