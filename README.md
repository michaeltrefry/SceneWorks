# SceneWorks

SceneWorks is a local Docker-based AI image and video generation studio. This repository currently contains a Vite/React web shell, Rust API backend, Rust utility worker, a native candle (CUDA) GPU inference worker, shared config/data folders, and Docker Compose wiring. (The legacy Python/PyTorch worker has been retired off-Mac — Docker GPU inference runs natively on candle; see Phase 7 / sc-5503.)

![SceneWorks](screenshot.png)

## Quick Start

```powershell
npm run dev
```

This starts the local stack with Docker Compose. Compose runs the Rust API and
Rust utility worker as the backend runtime, plus the candle (CUDA) GPU inference
worker for image and video generation (an NVIDIA GPU + the container toolkit are
required):

- Web: http://localhost:5173
- API: http://localhost:8000/api/v1/health

`GET /api/v1/health` reports `runtime: "rust"`. The Rust API image is built
from `docker/rust.Dockerfile (target rust-api)`; `SCENEWORKS_RUST_WORKER_GPU_ID=cpu` is the
default utility worker mode.

```powershell
docker compose build api
docker compose up -d api web worker
```

The API keeps the same Compose service name, health URL, worker URL, host port,
and mounted storage contracts. It listens on `SCENEWORKS_API_PORT` inside the
container and is exposed on the same host port.
`SCENEWORKS_WEB_PORT` controls the host port for the Vite web service. The web
service receives `VITE_API_BASE_URL=http://localhost:${SCENEWORKS_API_PORT}`,
and workers call `http://api:${SCENEWORKS_API_PORT}` on the compose network.
Compose builds the GPU inference worker from `docker/rust.Dockerfile (target
rust-worker-candle)` — the native candle/CUDA worker (an NVIDIA GPU + the NVIDIA
container toolkit are required). The API sets `SCENEWORKS_CANDLE_REQUIRED=1`, so a
job candle cannot serve fails with a precise, actionable error instead of waiting
forever; the Python torch worker is retired off-Mac (Phase 7 / sc-5503).

API volume contracts:

- `${SCENEWORKS_DATA_BIND:-./data}:/sceneworks/data` read/write for projects, models, LoRAs, and cache-backed app data.
- `${SCENEWORKS_CONFIG_BIND:-./config}:/sceneworks/config` writable for user manifests and app configuration.
- `./data/cache/jobs.db` is the queue database, preserving existing compose queue history across rebuilds.
- `./data/cache/huggingface` persists Diffusers/Hugging Face model downloads across worker container rebuilds and restarts.

The API exposes `GET /api/v1/health`; Compose checks it with `curl` inside the
container so dependent services wait for readiness. SceneWorks 0.2.0 queues
default clip duration payloads as JSON integers when `duration` is omitted or
integer-like, while explicit fractional values remain fractional.
To exercise the default Rust Docker path end to end, run:

```powershell
npm run check:docker:rust-api
```
Run the lightweight scaffold checks:

```powershell
npm run check
```

## Backend Runtime Split

The Rust backend workspace is the default Docker runtime. The Rust API owns the
HTTP surface and project/queue filesystem contracts, and the Rust worker owns
CPU utility jobs for model downloads, LoRA imports, FFmpeg frame extraction, and
timeline MP4 exports.

Install a Rust toolchain with `rustfmt` and `clippy`, then use:

```powershell
npm run rust:fmt
npm run rust:lint
npm run rust:test
npm run rust:build
```

Or run the full Rust verification sequence:

```powershell
npm run rust:check
```

Optionally install local Git hooks that run `npm run rust:fmt` before commits
touching Rust files:

```powershell
npm run hooks:install
```

To point host-mode workers at the API, start the Rust API binary on port 8000
and run each worker with `SCENEWORKS_API_URL=http://localhost:8000`. In Docker
Compose, workers are wired to the `api` service automatically. The
Compose `worker` service is the candle (CUDA) GPU inference worker and the
`rust-worker` service is the Rust utility worker; both use the same HTTP contract.
The `sceneworks-rust-worker` binary handles CPU utility jobs for model downloads,
LoRA imports, FFmpeg frame extraction, and timeline MP4 exports. The Rust utility
worker defaults to `SCENEWORKS_GPU_ID=cpu` and does not duplicate the candle GPU
worker. Utility jobs are I/O-bound and serialize per worker, so in cpu mode
it supervises a small pool of CPU utility workers (`SCENEWORKS_UTILITY_WORKERS`,
default 4) — this lets a quick upload run alongside a long download instead of
queueing behind it. Set it to `1` to restore single-worker behavior. Set
`SCENEWORKS_RUST_WORKER_GPU_ID=auto` in Compose, or `SCENEWORKS_GPU_ID=auto` in
host mode, only if you want the Rust worker to supervise one child per visible
NVIDIA GPU plus a CPU utility child; use `NVIDIA_VISIBLE_DEVICES=none` for a CPU
fallback-only worker or a comma-separated list to constrain GPU children.
Shutdown waits up to 10 seconds for child workers by default; set
`SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS` to tune that grace period. On
Windows, Rust listens for Ctrl+C; Unix workers also handle SIGTERM.

For the desktop build, the utility worker loop can run **inside** the API
process instead of as a separate binary: set `SCENEWORKS_RUN_UTILITY_INPROCESS=true`
and the API spawns the loop as a task that talks to the local API over loopback,
so a single `sceneworks-api` process serves the UI/API and claims utility jobs.
It honors the same `SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS` grace period on
Ctrl+C/SIGTERM. The Docker server leaves this `false` and runs the standalone
`sceneworks-rust-worker` container.

When running the stack outside Docker Compose, start `sceneworks-rust-worker`
alongside the API so Rust-owned utility jobs are claimed. GPU generation adapters
remain Python-owned: the Python worker advertises image/video generation and
person replacement capabilities on GPU children, backed by Diffusers/PyTorch.
Rust owns procedural person detection, person tracking, model, LoRA, and FFmpeg
utility families. The Python worker remains focused on Diffusers/PyTorch image
and video inference and no longer advertises or runs utility job fallbacks.
The Python worker ID changed from `worker-gpu-auto-0` to
`python-inference-worker-0`; existing queue databases may retain the old worker
row until the stale-worker sweep marks it offline.
When multiple GPU children are registered, auto GPU jobs may be claimed by a
worker that already reports the requested model as warm before falling back to
FIFO order. Explicit GPU selections and utility jobs keep their normal FIFO
claim order.
The Rust worker image installs Debian Bookworm `ffmpeg`; host-mode Rust workers
use the `ffmpeg` found on `PATH`. To download from gated/authenticated repos
(e.g. gated Hugging Face models, Civit.ai), add a token in the app — see
[Service Credentials](#service-credentials-api-tokens) below.

For the full per-job-kind breakdown of which worker (Python torch vs Rust MLX)
handles each job type — and the routing rules that decide — see the
[Worker Capability Matrix](crates/sceneworks-worker/ARCHITECTURE.md).

## Local Access Control

Local-only development is open by default. To require a simple pairing token for LAN or shared-machine use, copy `.env.example` to `.env` and set:

```text
SCENEWORKS_ACCESS_TOKEN=choose-a-private-token
```

When a token is configured, API requests other than health/access discovery must include either:

```text
Authorization: Bearer choose-a-private-token
```

or:

```text
X-SceneWorks-Token: choose-a-private-token
```

Event streams use a short-lived one-shot ticket instead of putting the access token in the URL. Clients should `POST /api/v1/jobs/events/ticket` with the normal auth header, then connect to `/api/v1/jobs/events?ticket=...`.

This is for privacy and control over local media, model downloads, and long-running GPU work. It is not a content moderation system.

The API binds to `127.0.0.1` (loopback) by default, so a direct binary run is not
reachable from the network until you opt in. To expose it (a server install, or
access from another machine), set `SCENEWORKS_API_HOST=0.0.0.0` **and** set
`SCENEWORKS_ACCESS_TOKEN` — without a token, every endpoint (project file reads,
credential writes, job creation, large model uploads) is reachable unauthenticated,
and the API logs a warning on startup. Inside Docker, `SCENEWORKS_API_HOST=0.0.0.0`
is set by `docker-compose.yml` so the published host port can reach the container,
but the host-side publish defaults to `SCENEWORKS_API_PUBLISH_HOST=127.0.0.1`.
Set `SCENEWORKS_API_PUBLISH_HOST=0.0.0.0` only when intentionally exposing Docker
Compose to the LAN; control access with `SCENEWORKS_ACCESS_TOKEN` and extend
`SCENEWORKS_CORS_ORIGINS` with LAN hostnames or IP origins when the web app is opened
from another machine.

For offline development or deterministic Rust API tests, set `SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE=1` to skip live Hugging Face model size lookups. The catalog still returns the same fields with unknown sizes.

## Service Credentials (API tokens)

Some model and LoRA downloads need an API token: gated Hugging Face repos (e.g.
FLUX.1 [dev]), Civit.ai, or any other authenticated source. SceneWorks stores
these as a generic, **host-keyed** credential — `{ host, label, scheme, token }`
— and attaches the matching one (as a `Bearer` header or a `?token=` query
parameter) when a download's URL host matches. Adding a new service needs no
code change.

**Where to add tokens:** open **Settings → Service credentials** and add the
host (e.g. `huggingface.co`), an optional label, the scheme (`bearer` or
`query`), and the token. Gated models on the **Models** screen show a notice
with a button that jumps straight here. Tokens are write-only — the UI never
displays a saved token again, only that one is present. Credential changes take
effect on the next worker restart.

**Where credentials live:**

- **Desktop:** the per-user OS keychain — Windows Credential Manager (DPAPI),
  macOS Keychain, or the Linux Secret Service. Nothing SceneWorks-managed holds
  an encryption key; the OS guards the secret per user account.
- **Server / Docker:** a `0600` JSON file, `credentials.json`, in the config
  dir (`/sceneworks/config` in Compose), managed over the authenticated REST API
  (`/api/v1/credentials`). There is no app-level encryption on the server — a key
  would have to live beside the data — so the protection is the restricted file
  mode plus your orchestrator's secret handling (Docker/Kubernetes secrets,
  host file permissions). Keep the config volume off shared/world-readable paths.

**Environment overrides (server):** the worker reads `credentials.json` and
overlays the optional `SCENEWORKS_CREDENTIALS` env var, a JSON map
`{ "host": { "token": "…", "scheme": "bearer|query" } }`; the env value **wins
per host**, so operators can inject secrets from a vault without writing the
file. Hugging Face keeps its dedicated path: set `HF_TOKEN` for gated HF repos
(the same variable `huggingface_hub` reads). Both are picked up at worker
startup.

## LoRA Training

SceneWorks can train LoRAs locally — image LoRAs for Z-Image-Turbo, Stable
Diffusion XL, and Microsoft Lens, plus video LoRAs for LTX-2.3 (Apple Silicon /
MLX) and Wan2.2 (CUDA *and* Apple Silicon): build a captioned dataset, validate
the plan with a dry run, train on a GPU worker, and the result is registered as
a normal SceneWorks LoRA — image LoRAs selectable in Image Studio, video LoRAs
in Video Studio. The Stable Diffusion XL kernel is the generic SDXL-UNet trainer
that SDXL-family models (e.g. Kolors) extend; the Wan2.2 A14B kernel trains both
mixture-of-experts denoisers (high/low-noise) as a per-expert LoRA pair.
See [documents/TRAINING_QUICKSTART.md](documents/TRAINING_QUICKSTART.md) for a
step-by-step first run, per-target notes, recommended dataset sizes and captions,
VRAM/disk notes, where outputs live, and troubleshooting. Training contracts live
in `crates/sceneworks-core/src/training.rs`; the execution kernel is
`apps/worker/scene_worker/training_adapters.py`.

## Structure

```text
apps/
  web/       React + Vite app shell
  rust-api/  Default Rust backend API
  rust-worker/ Rust CPU utility worker for model downloads, LoRA imports, frame extraction, and timeline exports
  worker/    Python Diffusers/PyTorch image and video inference sidecar
crates/
  sceneworks-core/ Shared Rust contract/domain helpers
packages/
  schemas/   Manifest JSON schemas
  shared/    Cross-app Python helpers for JSON, project lookup, and project DB indexing
config/
  manifests/ Built-in and user model/LoRA manifests
data/
  projects/  Local SceneWorks projects
  models/    App-managed model storage
  loras/     App-managed LoRA storage
  cache/     Runtime cache
docker/      Service Dockerfiles
```

## Desktop App (Tauri)

`apps/desktop` packages SceneWorks as a standalone desktop app (no Docker). It
bundles the Rust API and the builtin model/LoRA manifests, and runs generation on
a native, in-app engine — **MLX** on macOS (Apple Silicon) and **candle / CUDA**
on Windows (NVIDIA). There is no Python venv on either platform. On Windows, the
first run downloads the CUDA runtime into `%APPDATA%\SceneWorks\gpu-runtime`
(it is too large to ship in the installer), then starts the API + worker; macOS
runs the MLX engine in-process with no download step. Build the Windows installer
(NSIS) with:

```powershell
npm --prefix apps/desktop run build -- --bundles nsis
```

**For the end-user install, hardware requirements, storage layout, macOS GPU
memory tuning, supported features, and troubleshooting guide, see
[apps/desktop/README.md](apps/desktop/README.md).**

### Windows GPU support

SceneWorks on Windows requires an NVIDIA (CUDA) GPU with driver 576.02 or newer —
there is no CPU or AMD fallback. The bundled CUDA 12.9 runtime supports Blackwell
(sm_120).

| Tier | GPU | Status | Notes |
| --- | --- | --- | --- |
| High-end | NVIDIA RTX PRO 6000 Blackwell (96 GB) | ✅ Validated | Qwen image (~36 s incl. load), native LTX-2.3 text-to-video (~80 s for a 2 s clip), and timeline export verified end-to-end |
| Other CUDA | RTX 30/40-series, etc. | ⏳ Untested | Expected to work with driver ≥ 576.02; not yet validated |
| No NVIDIA GPU | — | ❌ Unsupported | Generation requires an NVIDIA GPU; no CPU/AMD fallback |

## Licensing

SceneWorks is a non-commercial, **source-available** project. Its own source
code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE): you
may use, modify, and share it freely for any **noncommercial** purpose, but
commercial use is not granted. (This is a source-available license, not an
OSI-approved "open source" license, because it restricts commercial use.) For
commercial licensing, contact the copyright holder.

**Model weights are not covered by this license.** SceneWorks downloads
third-party model weights at runtime, and each model keeps its own license —
some non-commercial (e.g. FLUX.1 [dev], FLUX.2 [klein] 9B), some permissive
(Apache-2.0 / OpenRAIL). You are responsible for complying with each model's
license when you download and use it; the SceneWorks license here applies only
to SceneWorks' own code, not to the weights it runs.

Bundled third-party source under `apps/*/scene_worker/_vendor/` is covered by
its own `LICENSE` files (Apache-2.0 / MIT), retained alongside that code.
