# SceneWorks Desktop

SceneWorks Desktop packages the full SceneWorks AI image and video studio as a
single native application — no Docker, no terminal, no Python to install. It is a
[Tauri](https://tauri.app) shell (`net.trefry.sceneworks`, v0.2.0) that bundles
the SceneWorks API and runs generation on a native, platform-specific engine:

- **macOS** runs on Apple's **MLX** engine (Apple Silicon only).
- **Windows** runs on the native **candle / CUDA** engine (NVIDIA only).

There is **no Python virtual environment** on either platform — the generation
engine is linked into the app, so first run starts straight on the native engine.

> **Looking to run SceneWorks as a server instead?** See
> [Desktop vs. server](#desktop-vs-server) below and the root
> [README](../../README.md) for the Docker Compose stack.

---

## Table of contents

- [Hardware requirements](#hardware-requirements)
- [Install](#install)
- [First run](#first-run)
- [Where your files live](#where-your-files-live)
- [GPU memory tuning (macOS)](#gpu-memory-tuning-macos)
- [Supported features & known limitations](#supported-features--known-limitations)
- [Desktop vs. server](#desktop-vs-server)
- [Troubleshooting](#troubleshooting)

---

## Hardware requirements

SceneWorks generation is GPU-only on both platforms. **There is no CPU or AMD
fallback** — a machine without a supported GPU can run the app shell but cannot
generate.

### Windows

| Requirement | Detail |
| --- | --- |
| OS | Windows 10 or 11 (64-bit) |
| GPU | NVIDIA, CUDA-capable |
| Driver | **576.02 or newer** (checked at first run; older drivers are blocked with a clear message) |
| VRAM | Per model class — see [VRAM by model](#vram-by-model). 12 GB runs most images; video and the larger image models want 24 GB+ |
| Disk | ~3 GB for the bundled GPU runtime downloaded on first run, plus model weights (tens of GB depending on what you pull) |

Blackwell (sm_120) GPUs are supported. The bundled CUDA 12.9 runtime forward-JITs
older PTX, so recent NVIDIA architectures work with a current driver.

### macOS

| Requirement | Detail |
| --- | --- |
| OS | macOS **26.2 or newer** |
| Chip | **Apple Silicon** (M-series). Max / Ultra tiers recommended for video |
| Unified memory | **64 GB minimum**; **96 GB+ recommended** for video and the largest image models |
| Disk | Model weights only (tens of GB depending on what you pull) — no GPU runtime download on macOS |

Intel Macs are not supported (no MLX). See
[GPU memory tuning (macOS)](#gpu-memory-tuning-macos) for getting the most out of
unified memory.

### VRAM by model

There is no fixed "VRAM per tier" number — each model declares its own memory
floor. In the app, open **Model Manager**: every model shows its estimated
download size and minimum memory. Under the hood these come from the built-in
catalog (`config/manifests/builtin.models.jsonc`, `limits.minMemoryGb`). As a
rough guide:

- **Images** range from lightweight SDXL-class models up to ~30–60 GB for the
  largest models (e.g. Lens, SenseNova U1 — the latter wants 96 GB+ unified
  memory on Apple Silicon).
- **Video** (LTX-2.3, Wan 2.2) is the most demanding path; budget 24 GB+ VRAM on
  Windows and 96 GB+ unified memory on Mac for comfortable headroom.

If a job needs more memory than the device has, it fails with a precise reason
rather than hanging — see [Troubleshooting](#troubleshooting).

---

## Install

### Windows

1. Download and run the SceneWorks installer (`.exe`, NSIS). It can install
   per-user or per-machine.
2. If Microsoft Edge **WebView2** is not already present, the installer fetches
   and installs it silently.
3. Launch **SceneWorks** from the Start menu.

The CUDA runtime is **not** in the installer (it is too large for the installer
format), so the first launch downloads it — see [First run](#first-run).

### macOS

1. Open the SceneWorks `.dmg` and drag **SceneWorks** to **Applications**.
2. Launch it. The app is signed with a Developer ID and notarized, so it opens
   without a Gatekeeper override.

---

## First run

On first launch a setup screen reports progress. What happens differs by platform.

### Windows (first run only)

1. **GPU preflight.** SceneWorks probes `nvidia-smi` for an NVIDIA GPU and driver
   version. With no GPU, or a driver below 576.02, setup stops here with a clear
   message — nothing is downloaded.
2. **GPU runtime download (~2.7 GB, once).** SceneWorks downloads the CUDA 12.9
   runtime, cuDNN, and the ONNX Runtime GPU provider into
   `%APPDATA%\SceneWorks\gpu-runtime`. Progress is shown per component. This is
   cached and version-marked, so later launches skip it. Time depends on your
   connection (typically a few minutes on a fast link).
3. **Engine starts.** The API starts and the candle/CUDA worker registers.

### macOS (first run)

There is no runtime download. The MLX engine is linked into the app, so it starts
the API and the MLX worker straight away — typically seconds.

### Both platforms

- The first-run splash lets you pick where your **data**, **config**, and
  **model cache** live (or accept the defaults below).
- **Model weights are not bundled.** Download the models you want from
  **Model Manager** inside the app. The first generation with a given model pulls
  its weights into the cache, so it is slower than subsequent runs.

---

## Where your files live

SceneWorks stores data under per-user OS locations. You can override the data and
config locations on the first-run splash (or later in **Settings**); the
environment variables in parentheses override them for advanced setups.

### Windows

| What | Location |
| --- | --- |
| Projects & app data | `%APPDATA%\SceneWorks\data` (`SCENEWORKS_DATA_DIR`) |
| Config & manifests | `%APPDATA%\SceneWorks\config` (`SCENEWORKS_CONFIG_DIR`) |
| Cache | `%LOCALAPPDATA%\SceneWorks\cache` |
| Logs | `%LOCALAPPDATA%\SceneWorks\logs` |
| GPU runtime | `%APPDATA%\SceneWorks\gpu-runtime` |
| Model weights | `%USERPROFILE%\.cache\huggingface` (`HF_HOME`) |

### macOS

| What | Location |
| --- | --- |
| Projects & app data | `~/Library/Application Support/SceneWorks/data` (`SCENEWORKS_DATA_DIR`) |
| Config & manifests | `~/Library/Application Support/SceneWorks/config` (`SCENEWORKS_CONFIG_DIR`) |
| Cache | `~/Library/Caches/SceneWorks` |
| Logs | `~/Library/Logs/SceneWorks` |
| Model weights | `~/.cache/huggingface` (`HF_HOME`) |

Model weights default to the shared Hugging Face cache so SceneWorks reuses
anything already downloaded by other tools on the machine (and vice versa). Point
`HF_HOME` (or the splash field) at a larger drive if your boot volume is tight.

Service credentials (gated Hugging Face / Civitai tokens) are **not** stored in
these folders — they go in the per-user OS keychain (Windows Credential Manager,
macOS Keychain). Add them under **Settings → Service credentials**.

---

## GPU memory tuning (macOS)

On Apple Silicon, CPU and GPU share one pool of unified memory. macOS caps how
much of that pool the GPU may "wire" (hold resident). For the largest models —
especially video on 64 GB machines — raising that cap can be the difference
between a job running and a job failing on allocation.

The current cap is shown in **Settings** (alongside total unified memory).
SceneWorks reads it but does not change it. To raise it yourself:

```sh
# Example: allow the GPU to wire up to ~56 GB on a 64 GB machine.
sudo sysctl iogpu.wired_limit_mb=57344
```

Pick a value in MB that leaves headroom for macOS and other apps. A common
starting point is ~85–90% of total unified memory:

- 64 GB → try `57344` (56 GB)
- 96 GB → try `86016` (84 GB)
- 128 GB → try `114688` (112 GB)

**Tradeoffs and cautions:**

- Setting it too high starves macOS and other apps; the system can become
  unstable or swap heavily. Increase gradually and watch memory pressure in
  Activity Monitor.
- The setting is **not persistent** — it resets on reboot. Re-run the command, or
  add it to a launch daemon if you want it to stick.
- This only helps when the GPU was being throttled by the wired limit; it does not
  create memory that isn't there. If a model's floor exceeds your total unified
  memory, more weights/quantization or a smaller model is the answer.

---

## Supported features & known limitations

SceneWorks advertises capabilities per device, so the UI only offers what the
current machine can actually run.

| Capability | macOS (MLX) | Windows (candle / CUDA) |
| --- | --- | --- |
| Image generate / edit / detail | ✅ | ✅ |
| Image VQA / interleave | ✅ | ✅ |
| Video generate / extend / bridge | ✅ | ✅ |
| Person replace (VACE) | ✅ (native MLX `wan_vace`) | ✅ (candle; legacy torch CUDA path) |
| Pose / keypoint / person detect & track | ✅ | ✅ |
| Image / video upscale | ✅ | ✅ |
| Image / person **segmentation** | ✅ | ⏳ not yet ported off-Mac |
| LoRA training (image & video) | ✅ | ✅ (see Training Quickstart) |

Notes:

- **Person replacement (VACE).** On Mac this runs end-to-end on the native MLX
  `wan_vace` provider (validated on Apple Silicon, including character-LoRA runs).
  On Windows it runs on the candle/CUDA path. The two engines are feature-parity
  for replace-person.
- **Segmentation** has not been ported to the candle worker yet; segmentation-based
  features are Mac-only for now.
- **LoRA training** is supported on both platforms. See
  [TRAINING_QUICKSTART.md](../../documents/TRAINING_QUICKSTART.md) for per-target
  notes, dataset sizes, and VRAM/disk guidance.

### Windows GPU validation

| Tier | GPU | Status | Notes |
| --- | --- | --- | --- |
| High-end | NVIDIA RTX PRO 6000 Blackwell (96 GB) | ✅ Validated | Full happy path end-to-end: Qwen image (~36 s incl. load), native LTX-2.3 text-to-video (~80 s for a 2 s 768×512 clip), and 720p timeline export |
| Other CUDA | RTX 30/40-series, etc. | ⏳ Untested | Expected to work with driver ≥ 576.02; not yet formally validated |
| No NVIDIA GPU | — | ❌ Unsupported | Generation requires an NVIDIA GPU; no CPU/AMD fallback |

Performance numbers are indicative of the validated hardware and will vary by GPU,
model, and settings.

### macOS tier validation

Apple Silicon tiers are validated by unified-memory size; each model declares its
own floor (see [VRAM by model](#vram-by-model)). On this hardware the demanding MLX
paths (LTX-2.3 Q4, Wan 2.2 5B, large images) settle at a **~50 GB** working set —
they fit 64 GB, but with thin headroom, so raise the
[wired limit](#gpu-memory-tuning-macos) and close other apps for reliable runs.

| Tier | Example chip | Status | Notes |
| --- | --- | --- | --- |
| 64 GB | Apple M5 Max | ✅ Validated (minimum) | Happy path end-to-end: Z-Image-Turbo image (1024², ~68 s cold), LTX-2.3 MLX Q4 text-to-video (2 s/768×512, ~41 s), Wan 2.2 5B text-to-video (~2.7 min), 720p timeline export (~2 s). Demanding paths peak ~50 GB unified. **Qwen-Image does not currently work on Mac (a known tokenizer issue) — use Z-Image-Turbo instead.** Wan 2.2 14B (133 GB floor) and person-replace/VACE (96 GB floor) are **not available at this tier.** |
| 96 GB | — | ⏳ Recommended, not yet validated | Comfortable headroom for video and the largest image models |
| 128 GB | Apple M-series (≈128 GB+) | ⏳ Partially characterized | MLX benchmarks: LTX-2.3 Q4 ~37.5 s, Wan 2.2 5B ~4.9 min (40-step), Wan 2.2 14B + LoRA ~4.3 min |

Performance numbers are indicative of the validated hardware and will vary by chip,
model, and settings.

---

## Desktop vs. server

SceneWorks ships in two forms from the same codebase:

| | **Desktop** (this app) | **Server** (Docker) |
| --- | --- | --- |
| Install | One installer, no Docker | `docker compose` stack |
| Engine | MLX (Mac) / candle CUDA (Windows), in-app | candle CUDA worker container (NVIDIA + container toolkit) |
| Access | Local app window, loopback only | Web UI over the network (opt-in, token-gated) |
| Credentials | OS keychain | `0600 credentials.json` / env vars |
| Paths | Per-user OS app-data dirs | Fixed `/sceneworks/*` volumes |
| Best for | A single creator on one workstation | Shared/remote GPUs, multiple users, headless hosts |

**Use the desktop app** when you want a turnkey studio on your own Windows or Mac
workstation. **Use the server stack** when the GPU lives on another machine, when
several people share it, or when you want a headless/LAN deployment. The Docker
stack lives at the repo root (`docker-compose.yml`); see the root
[README](../../README.md) for setup and access-control details.

---

## Troubleshooting

### Where the logs are

| Platform | Logs directory |
| --- | --- |
| macOS | `~/Library/Logs/SceneWorks/` |
| Windows | `%LOCALAPPDATA%\SceneWorks\logs\` |

Key files: `api.log` (the API), and the engine worker log — `mlx-worker.log` on
macOS, `candle-worker.log` on Windows. The current session's logs are also visible
in-app on the **Logs** screen.

### Common issues

**Windows: "SceneWorks on Windows requires an NVIDIA (CUDA) GPU."**
No NVIDIA GPU was detected. SceneWorks generation is CUDA-only on Windows — there
is no CPU or AMD fallback. A supported NVIDIA GPU is required.

**Windows: "requires NVIDIA driver 576.02 or newer."**
Your driver is below the floor for the bundled CUDA 12.9 runtime. Update your
NVIDIA driver from the GeForce/Studio app or nvidia.com, then relaunch.

**Windows: "GPU runtime download failed."**
The first-run CUDA runtime download did not complete (network/disk). Check your
connection and free space, then retry from the setup screen. The download resumes
into `%APPDATA%\SceneWorks\gpu-runtime`.

**"The local API did not start in time."**
The bundled API didn't become healthy within the startup window. Retry from the
setup screen; if it persists, check `api.log` in the logs directory above.

**A model out-of-memories (OOM) / a job fails with a memory reason.**
The model's memory floor exceeds what's available. Options:
- Choose a smaller or more heavily quantized model in **Model Manager** (model
  cards show the minimum memory).
- On macOS, raise the GPU wired limit — see
  [GPU memory tuning (macOS)](#gpu-memory-tuning-macos).
- Close other GPU-heavy apps to free memory.

Because each engine enforces what it can actually serve, an unservable job fails
**loudly with a precise reason** rather than hanging — check the worker log for
the exact cause.

**A Mac video / replace-person job seems stuck "queued."**
The MLX worker claims one job at a time; a long job in flight holds the slot.
Check `mlx-worker.log` — if nothing is being claimed, the worker may not be idle.

**Gated model won't download (401 / auth error).**
Add the service token under **Settings → Service credentials** (e.g.
`huggingface.co` for gated Hugging Face repos). Credential changes take effect on
the next worker restart.
