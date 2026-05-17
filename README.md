# SceneWorks

SceneWorks is a local Docker-based AI image and video generation studio. This repository currently contains a Vite/React web shell, FastAPI backend, Python inference worker, Rust utility worker, shared config/data folders, and Docker Compose wiring.

## Quick Start

```powershell
npm run dev
```

This starts the local stack with Docker Compose:

- Web: http://localhost:5173
- API: http://localhost:8000/api/v1/health

Run the lightweight scaffold checks:

```powershell
npm run check
```

## Rust Backend Migration

The Rust backend workspace is scaffolded for the migration spine, but it is not
wired into the default Docker runtime yet. The existing FastAPI API, Python
worker, and React app remain the default development stack.

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

To point workers at the Rust API during migration testing, start the Rust API
binary on port 8000 and run the worker with `SCENEWORKS_API_URL=http://localhost:8000`.
The `sceneworks-rust-worker` binary handles CPU utility jobs for model downloads
and LoRA imports.

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

For offline development or deterministic Rust API tests, set `SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE=1` to skip live Hugging Face model size lookups. The catalog still returns the same fields with unknown sizes.

## Structure

```text
apps/
  web/       React + Vite app shell
  api/       FastAPI service and backend filesystem owner
  rust-api/  Rust backend migration scaffold, not in the default runtime
  rust-worker/ Rust CPU utility worker for model downloads and LoRA imports
  worker/    Placeholder worker package
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
