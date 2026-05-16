from functools import lru_cache
import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.app_version = os.getenv("SCENEWORKS_APP_VERSION", "0.1.0")
        self.host = os.getenv("SCENEWORKS_API_HOST", "0.0.0.0")
        self.port = int(os.getenv("SCENEWORKS_API_PORT", "8000"))
        self.data_dir = Path(os.getenv("SCENEWORKS_DATA_DIR", "data")).resolve()
        self.config_dir = Path(os.getenv("SCENEWORKS_CONFIG_DIR", "config")).resolve()
        self.access_token = os.getenv("SCENEWORKS_ACCESS_TOKEN", "").strip()
        cors = os.getenv(
            "SCENEWORKS_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        self.cors_origins = [origin.strip() for origin in cors.split(",") if origin.strip()]

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "recent-projects.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
