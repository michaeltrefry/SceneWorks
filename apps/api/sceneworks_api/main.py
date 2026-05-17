from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .assets import router as assets_router
from .characters import router as characters_router
from .events import EventHub, EventTicketStore
from .image_generation import router as image_router
from .jobs import router as jobs_router
from .jobs_store import JobsStore
from .models import loras_router, router as models_router
from .person_tracking import router as person_tracking_router
from .projects import ensure_data_dirs, router as projects_router
from .security import access_control_middleware
from .settings import Settings, get_settings
from .timelines import router as timelines_router
from .video_generation import router as video_router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    ensure_data_dirs(settings)
    jobs_store = JobsStore(settings.jobs_db_path)
    jobs_store.initialize()
    interrupted_jobs = jobs_store.mark_interrupted_on_startup()

    app = FastAPI(
        title="SceneWorks API",
        version=settings.app_version,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.jobs_store = jobs_store
    app.state.event_hub = EventHub()
    app.state.event_ticket_store = EventTicketStore()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-SceneWorks-Token"],
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
                "jobsDb": str(settings.jobs_db_path),
            },
            "interruptedJobsOnStartup": len(interrupted_jobs),
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

    app.include_router(projects_router, prefix="/api/v1")
    app.include_router(assets_router, prefix="/api/v1")
    app.include_router(characters_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(loras_router, prefix="/api/v1")
    app.include_router(timelines_router, prefix="/api/v1")
    app.include_router(person_tracking_router, prefix="/api/v1")
    app.include_router(image_router, prefix="/api/v1")
    app.include_router(video_router, prefix="/api/v1")
    app.include_router(jobs_router, prefix="/api/v1")
    return app


app = create_app()
