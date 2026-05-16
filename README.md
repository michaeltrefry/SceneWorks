# SceneWorks

SceneWorks is a local Docker-based AI image and video generation studio. This repository currently contains the runtime skeleton for the first epic: a Vite/React web shell, FastAPI backend, placeholder Python worker, shared config/data folders, and Docker Compose wiring.

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

This is for privacy and control over local media, model downloads, and long-running GPU work. It is not a content moderation system.

## Structure

```text
apps/
  web/       React + Vite app shell
  api/       FastAPI service and backend filesystem owner
  worker/    Placeholder worker package
packages/
  schemas/   Shared schema placeholders
  shared/    Cross-app shared notes/helpers placeholder
config/
  manifests/ Built-in and user model/LoRA manifests
data/
  projects/  Local SceneWorks projects
  models/    App-managed model storage
  loras/     App-managed LoRA storage
  cache/     Runtime cache
docker/      Service Dockerfiles
```
