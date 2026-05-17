# SceneWorks Python API

FastAPI rollback service for versioned backend routes and project filesystem
writes. The default Docker Compose backend is the Rust API; select this service
only by setting `SCENEWORKS_API_RUNTIME=python` and
`SCENEWORKS_API_DOCKERFILE=docker/api.Dockerfile`.

Current routes:

- `GET /api/v1/health`
- `GET /api/v1/access`
- `POST /api/v1/auth/verify`
- `GET /api/v1/projects`
- `POST /api/v1/projects`
- `GET /api/v1/projects/{project_id}`
- `POST /api/v1/projects/{project_id}/reindex`
- `POST /api/v1/jobs/events/ticket`
- `GET /api/v1/jobs/events`
