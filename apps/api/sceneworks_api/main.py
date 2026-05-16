from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .projects import ensure_data_dirs, router as projects_router
from .security import access_control_middleware
from .settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    ensure_data_dirs(settings)

    app = FastAPI(
        title="SceneWorks API",
        version=settings.app_version,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _access_control(request: Request, call_next):
        return await access_control_middleware(request, call_next, settings)

    @app.get("/api/v1/health", tags=["system"])
    def health() -> dict:
        return {
            "status": "ok",
            "service": "sceneworks-api",
            "version": settings.app_version,
            "authRequired": bool(settings.access_token),
            "directories": {
                "data": str(settings.data_dir),
                "config": str(settings.config_dir),
                "projects": str(settings.projects_dir),
            },
        }

    @app.get("/api/v1/access", tags=["system"])
    def access() -> dict:
        return {
            "authRequired": bool(settings.access_token),
            "tokenHeader": "X-SceneWorks-Token",
        }

    @app.post("/api/v1/auth/verify", tags=["system"])
    def verify_access(request: Request) -> dict:
        from .security import is_authorized

        return {"ok": is_authorized(request, settings)}

    @app.get("/api/v1/jobs/events", tags=["jobs"])
    async def job_events() -> StreamingResponse:
        async def stream():
            yield "event: ready\n"
            yield 'data: {"status":"placeholder","message":"Job event stream ready"}\n\n'

        return StreamingResponse(stream(), media_type="text/event-stream")

    app.include_router(projects_router, prefix="/api/v1")
    return app


app = create_app()
