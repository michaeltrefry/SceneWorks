# SceneWorks

SceneWorks is a local Docker-based AI image and video generation studio. This repository currently contains a Vite/React web shell, Rust API backend, Rust utility worker, Python Diffusers/PyTorch inference worker, shared config/data folders, and Docker Compose wiring.

## Quick Start

```powershell
npm run dev
```

This starts the local stack with Docker Compose. Compose runs the Rust API and
Rust utility worker as the backend runtime, plus the Python worker only for
Diffusers/PyTorch image and video inference adapters:

- Web: http://localhost:5173
- API: http://localhost:8000/api/v1/health

`GET /api/v1/health` reports `runtime: "rust"`. The Rust API image is built
from `docker/rust-api.Dockerfile`; `SCENEWORKS_RUST_WORKER_GPU_ID=cpu` is the
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
Compose builds the Python inference worker with CUDA PyTorch wheels by default
using `SCENEWORKS_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`.
Set `SCENEWORKS_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` before
building only when you intentionally want a CPU-only worker; CPU-only PyTorch
workers do not advertise image or video inference capabilities.

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

To point host-mode workers at the API, start the Rust API binary on port 8000
and run each worker with `SCENEWORKS_API_URL=http://localhost:8000`. In Docker
Compose, workers are wired to the `api` service automatically. The
Compose `worker` service is the Python inference worker and the `rust-worker`
service is the Rust utility worker; both use the same HTTP contract.
The `sceneworks-rust-worker` binary handles CPU utility jobs for model downloads,
LoRA imports, FFmpeg frame extraction, and timeline MP4 exports. The Rust worker
defaults to `SCENEWORKS_GPU_ID=cpu` and does not duplicate the Python inference
GPU workers. Utility jobs are I/O-bound and serialize per worker, so in cpu mode
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
use the `ffmpeg` found on `PATH`. Set `HF_TOKEN` when downloading from gated
Hugging Face repositories.

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

Inside Docker, `SCENEWORKS_API_HOST=0.0.0.0` is expected so the published host
port can reach the container. Use `SCENEWORKS_ACCESS_TOKEN` for access control
and extend `SCENEWORKS_CORS_ORIGINS` with LAN hostnames or IP origins when the
web app is opened from another machine.

For offline development or deterministic Rust API tests, set `SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE=1` to skip live Hugging Face model size lookups. The catalog still returns the same fields with unknown sizes.

## LoRA Training

SceneWorks can train an image LoRA for Z-Image-Turbo locally: build a captioned
dataset, validate the plan with a dry run, train on a CUDA GPU worker, and the
result is registered as a normal SceneWorks LoRA selectable in Image Studio.
See [documents/TRAINING_QUICKSTART.md](documents/TRAINING_QUICKSTART.md) for a
step-by-step first run, recommended dataset sizes and captions, VRAM/disk notes,
where outputs live, and troubleshooting. Training contracts live in
`crates/sceneworks-core/src/training.rs`; the execution kernel is
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
  schemas/   Shared schema placeholders
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
bundles the Rust API, the Python worker source (`scene_worker` +
`sceneworks_shared`), the builtin model/LoRA manifests, and a pinned `uv`. On
first run it provisions a CUDA-enabled Python venv under the per-user app data
dir (`%APPDATA%\SceneWorks` on Windows), then starts the API + worker. Build the
Windows installer (NSIS) with:

```powershell
npm --prefix apps/desktop run build -- --bundles nsis
```

First-run installs CUDA torch wheels from `cu128` by default (override with
`SCENEWORKS_PYTORCH_INDEX_URL`). Blackwell (sm_120) requires `cu128`; older
`cu121` wheels will not run on it.

### Windows GPU support

| Tier | GPU | Status | Notes |
| --- | --- | --- | --- |
| High-end | NVIDIA RTX PRO 6000 Blackwell (96 GB) | ✅ Validated | torch 2.8 `cu128`; Qwen image (~36 s incl. load), native LTX-2.3 text-to-video (~80 s for a 2 s clip), and timeline export verified end-to-end |
| Other CUDA | RTX 30/40-series, etc. | ⏳ Untested | Expected to work via the `cu128` wheels with a recent driver; not yet validated |
| CPU-only | — | ⚠️ Limited | No GPU inference; image/video generation unavailable |
