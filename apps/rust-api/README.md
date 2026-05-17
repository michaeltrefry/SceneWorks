# SceneWorks Rust API

This binary is the default SceneWorks backend API. Docker Compose runs it as the
`api` service with these defaults:

```text
SCENEWORKS_API_RUNTIME=rust
SCENEWORKS_API_DOCKERFILE=docker/rust-api.Dockerfile
```

The Python FastAPI service remains the rollback runtime:

```text
SCENEWORKS_API_RUNTIME=python
SCENEWORKS_API_DOCKERFILE=docker/api.Dockerfile
```

Both API runtimes use the same compose contracts:

- `SCENEWORKS_API_HOST` and `SCENEWORKS_API_PORT` control the container bind address.
- `SCENEWORKS_DATA_DIR=/sceneworks/data` maps to `${SCENEWORKS_DATA_BIND:-./data}`.
- `SCENEWORKS_CONFIG_DIR=/sceneworks/config` maps read-only to `${SCENEWORKS_CONFIG_BIND:-./config}`.
- `SCENEWORKS_JOBS_DB_PATH=/sceneworks/data/cache/jobs.db` stores queue state on the existing data bind mount.
- `SCENEWORKS_ACCESS_TOKEN`, `SCENEWORKS_CORS_ORIGINS`, and `SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE` are honored by the Rust API.

Compose checks `GET /api/v1/health` inside the container before starting
dependent services. The health payload includes `runtime` so migration checks
can confirm which implementation is serving traffic.

After changing `SCENEWORKS_API_DOCKERFILE`, rebuild the API image before
starting it:

```powershell
docker compose build api
docker compose up -d api
```

Use the root Rust scripts to format, lint, test, and build this workspace.
