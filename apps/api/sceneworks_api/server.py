import uvicorn

from .settings import Settings


def run(settings: Settings) -> None:
    uvicorn.run(
        "sceneworks_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
