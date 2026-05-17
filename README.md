# SceneWorks

SceneWorks is a local Docker-based AI image and video generation studio. This repository currently contains a Vite/React web shell, Rust API backend, Rust utility worker, Python Diffusers/PyTorch inference worker, shared config/data folders, and Docker Compose wiring.

## Quick Start

```powershell
npm run dev
```

This starts the local stack with Docker Compose. By default, Compose now runs
the Rust API and Rust utility worker as the backend runtime, plus the Python
worker only for Diffusers/PyTorch image and video inference adapters:

- Web: http://localhost:5173
- API: http://localhost:8000/api/v1/health

The selected API runtime is reported by `GET /api/v1/health` as `runtime`.
Default Compose values are:

```text
SCENEWORKS_API_RUNTIME=rust
SCENEWORKS_API_DOCKERFILE=docker/rust-api.Dockerfile
SCENEWORKS_PYTHON_UTILITY_JOBS=1
```

Rollback to the Python API remains available by setting these values in `.env`
before rebuilding and starting the stack:

```text
SCENEWORKS_API_RUNTIME=python
SCENEWORKS_API_DOCKERFILE=docker/api.Dockerfile
```

```powershell
docker compose build api
docker compose up -d api web worker
```

Both API images keep the same Compose service name, health URL, worker URL,
host port, and mounted storage contracts. The API listens on
`SCENEWORKS_API_PORT` inside the container and is exposed on the same host port.
`SCENEWORKS_WEB_PORT` controls the host port for the Vite web service. The web
service receives `VITE_API_BASE_URL=http://localhost:${SCENEWORKS_API_PORT}`,
and workers call `http://api:${SCENEWORKS_API_PORT}` on the compose network.

API volume contracts are shared across Python and Rust:

- `${SCENEWORKS_DATA_BIND:-./data}:/sceneworks/data` read/write for projects, models, LoRAs, and cache-backed app data.
- `${SCENEWORKS_CONFIG_BIND:-./config}:/sceneworks/config:ro` read-only for manifests and app configuration.
- `./data/cache/jobs.db` is the shared queue database for both runtimes, preserving existing compose queue history across migration flips and rollback.

Both API runtimes expose `GET /api/v1/health`; Compose checks it with `curl`
inside the container so dependent services wait for the selected implementation.
SceneWorks 0.2.0 also aligns Rust video job payloads with the Python wire
shape for default clip duration: omitted or integer `duration` values are queued
as JSON integers, while explicit fractional values remain fractional.
To exercise the default Rust Docker path end to end, run:

```powershell
npm run check:docker:rust-api
```
To exercise the Python API rollback path, run:

```powershell
npm run check:docker:python-api
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
Compose, workers are wired to the selected `api` service automatically. The
Compose `worker` service is the Python inference worker and the `rust-worker`
service is the Rust utility worker; both use the same HTTP contract.
The `sceneworks-rust-worker` binary handles CPU utility jobs for model downloads,
LoRA imports, FFmpeg frame extraction, and timeline MP4 exports. Set
`SCENEWORKS_GPU_ID=auto` to let the Rust worker supervise one child per visible
NVIDIA GPU plus a CPU utility child; use `NVIDIA_VISIBLE_DEVICES=none` for a CPU
fallback-only worker or a comma-separated list to constrain the GPU children.
The Rust worker defaults to auto mode, matching the Python worker. Shutdown waits
up to 10 seconds for child workers by default; set
`SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS` to tune that grace period. On
Windows, Rust listens for Ctrl+C; Unix workers also handle SIGTERM.

When running the stack outside Docker Compose, start `sceneworks-rust-worker`
alongside the API so Rust-owned utility jobs are claimed. GPU generation adapters
remain Python-owned: the Python worker advertises image/video generation and
person replacement capabilities on GPU children, backed by Diffusers/PyTorch.
Compose sets `SCENEWORKS_PYTHON_UTILITY_JOBS=1` so procedural person detection
and person tracking jobs continue to be claimed by the Python worker while Rust
owns the model, LoRA, and FFmpeg utility families. Set
`SCENEWORKS_LEGACY_MODEL_LORA_JOBS=1` only when temporarily rolling
`model_download` or `lora_import` back to Python, and set
`SCENEWORKS_LEGACY_FFMPEG_JOBS=1` only when rolling `frame_extract` or
`timeline_export` back to Python.
The Python worker ID changed from `worker-gpu-auto-0` to
`python-inference-worker-0`; existing queue databases may retain the old worker
row until the stale-worker sweep marks it offline.
Both Docker worker images install Debian Bookworm `ffmpeg`; host-mode workers
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

## Structure

```text
apps/
  web/       React + Vite app shell
  api/       FastAPI API rollback runtime
  rust-api/  Default Rust backend API
  rust-worker/ Rust CPU utility worker for model downloads, LoRA imports, frame extraction, and timeline exports
  worker/    Python Diffusers/PyTorch inference worker and documented utility fallback
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
