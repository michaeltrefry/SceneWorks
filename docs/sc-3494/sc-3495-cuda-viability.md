# sc-3495 — Candle CUDA build/runtime viability on Windows: PASS

Epic 3494 (Candle SDXL Windows feasibility spike). Outcome: **Candle builds and
runs a CUDA workload on this Windows/Blackwell box.** A minimal Candle smoke
allocated a tensor on the Blackwell GPU (sm_120) and ran a real matmul+softmax
kernel correctly. The SceneWorks Rust worker can reasonably link Candle CUDA on
Windows, subject to the MSVC-toolset constraint documented below.

## Result

```
[build] candle-core git main; cuda feature = true
[device] CUDA available -> using cuda:0
[alloc]  2x 1024x1024 f32 tensors in 8.0 ms
[op]     matmul+softmax+sum in 7837 ms   (cold start: one-time cuBLAS + sm_120 kernel JIT)
[check]  sum(softmax(matmul)) = 1024.000 (expected ~= 1024.0)
[result] PASS
```
The 7.8 s op time is first-call CUDA context / cuBLAS / PTX-JIT warmup, not
steady-state throughput. Numeric result is exact, confirming correct execution.

## Environment (captured)

| Item | Value |
|---|---|
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell, compute cap **12.0 (sm_120)**, 96 GB |
| Driver | 596.36 |
| OS | Windows 11 Pro (26200) |
| Rust | 1.95.0, host `x86_64-pc-windows-msvc` |
| Candle | `candle-core` 0.10.2, git main @ `3d3d9c41c5bcfeb0e18b269ba938bb498575472a` |
| cudarc | 0.19.7 (`cuda-version-from-build-system`) |
| CUDA Toolkit | **12.9** (nvcc V12.9.41) |
| Host compiler | **MSVC 14.44.35207** (VS 2022 Build Tools, v143) |
| `CUDA_COMPUTE_CAP` | `120` |

## The one real gotcha: MSVC toolset version

CUDA 12.9's `nvcc` only accepts a VS 2017–2022 host compiler. This machine's
default is **Visual Studio 2026, MSVC 14.51 ("v18")**, which fails two ways:
1. `host_config.h(170) C1189: unsupported Microsoft Visual Studio version!`
2. With `--allow-unsupported-compiler` the gate is bypassed but compilation then
   fails *inside MSVC 14.51's own STL* (`type_traits`, `utility`, `yvals_core.h`)
   — nvcc 12.9's front-end cannot parse the 14.51 standard library.

**Fix:** install the **VS 2022 Build Tools v143 toolset (MSVC 14.44)** and build
from *its* `vcvars64`, so `cl.exe` and `INCLUDE` resolve to 14.44. No
`--allow-unsupported-compiler` needed. (Until CUDA raises its host-compiler
ceiling to VS 2026, this side-by-side toolset is mandatory on this box.)

## Setup recipe (reproducible)

Prereqs (one-time, admin):
1. CUDA Toolkit 12.9 (Development + VS integration). Sets `CUDA_PATH`, `nvcc`.
   - Verify: `nvcc --list-gpu-code` lists `sm_120`.
2. VS 2022 Build Tools with VCTools workload (MSVC v143) + Windows 11 SDK:
   ```
   winget install --id Microsoft.VisualStudio.2022.BuildTools -e ^
     --override "--quiet --wait --norestart ^
       --add Microsoft.VisualStudio.Workload.VCTools ^
       --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
       --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"
   ```

Build/run (see `build_cuda.ps1`):
```powershell
# 1. import the VS 2022 v143 env (NOT VS 2026):
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
# 2. point at CUDA 12.9 + pin Blackwell cap:
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:CUDA_COMPUTE_CAP = "120"
# 3. build with the cuda feature:
cargo build --release --features cuda
```

### Notes / DLL & path
- `candle-core/cuda` → `cudarc` with `dynamic-linking`: CUDA DLLs (`cudart64`,
  `cublas64`, `nvrtc64`, `curand64`) are loaded at runtime from `CUDA_PATH\bin`,
  which must be on `PATH` for the worker process. No DLL renaming needed.
- `CUDA_COMPUTE_CAP=120` is set explicitly; candle would otherwise auto-detect it
  from `nvidia-smi --query-gpu=compute_cap` (→ `12.0` → `120`). Explicit is safer
  for headless/service contexts where `nvidia-smi` may not be on PATH.
- First-run latency: expect multi-second cold start for CUDA context + cuBLAS +
  PTX JIT on sm_120; amortized after first op.

## Packaging implications (native vs Docker/WSL)
- **Native Windows worker:** ship against `CUDA_PATH\bin`; ensure the v143 toolset
  is the *build* host (build-time only — not a runtime dependency). Runtime needs
  only the CUDA runtime DLLs + NVIDIA driver (already present via `nvcuda.dll`).
- **Docker/WSL:** would use a Linux CUDA base image and gcc host compiler, side-
  stepping the MSVC-version issue entirely; separate path, not validated here.

## Distribution / multi-GPU packaging (build-time vs runtime)

A binary built with `CUDA_COMPUTE_CAP=120` contains kernels for **Blackwell sm_120
only** and will fail on other GPUs ("no kernel image available"). This is a build
choice, NOT per-machine compilation:

- **Fat binary:** nvcc can embed SASS for many arches + PTX for the newest in one
  `.exe` (how PyTorch/ComfyUI/ollama ship). Build a cap list (e.g.
  `sm_80;86;89;90;120` + PTX) → one worker binary runs on all targeted GPUs; the
  driver JITs PTX for any newer arch. Cost: longer compile + bigger binary. You
  compile **once in CI**, never on customer hardware. (Verify candle's build script
  accepts a multi-cap list cleanly — single-arch proven, multi-arch is a follow-up.)
- **Build-time only:** CUDA Toolkit (nvcc) + the v143 MSVC toolset live on the CI
  build box. End users install neither.
- **Runtime needs:** (1) an NVIDIA **driver** new enough for the CUDA 12.x runtime +
  the GPU arch (document a floor; build box is 596.36) — not pinned to one driver;
  (2) the CUDA **runtime redistributable DLLs** (`cudart`, `cublas`, `cublasLt`,
  `curand`, `nvrtc`) bundled with the worker (cudarc uses dynamic-linking), exactly
  like a torch wheel bundles its CUDA libs.

Net: same distribution model the current Python/torch worker already uses (multi-arch
+ bundled CUDA libs) — a Rust binary + CUDA redistributables instead of a Python env.

## Verdict for the worker
Linking Candle CUDA into the SceneWorks Rust worker on Windows is **viable**. The
only non-obvious requirement is pinning a CUDA-supported MSVC toolset (v143) for
the build; everything else (crate resolution, MSVC link, sm_120 kernel codegen,
runtime execution) works. Cleared to proceed to the SDXL prototype (sc-3497).
```
