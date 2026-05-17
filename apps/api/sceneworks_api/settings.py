from functools import lru_cache
import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.api_runtime = os.getenv("SCENEWORKS_API_RUNTIME", "python").strip() or "python"
        self.app_version = os.getenv("SCENEWORKS_APP_VERSION", "0.2.0")
        self.host = os.getenv("SCENEWORKS_API_HOST", "0.0.0.0")
        self.port = int(os.getenv("SCENEWORKS_API_PORT", "8000"))
        self.data_dir = Path(os.getenv("SCENEWORKS_DATA_DIR", "data")).resolve()
        self.config_dir = Path(os.getenv("SCENEWORKS_CONFIG_DIR", "config")).resolve()
        self.access_token = os.getenv("SCENEWORKS_ACCESS_TOKEN", "").strip()
        self.worker_timeout_seconds = int(os.getenv("SCENEWORKS_WORKER_TIMEOUT_SECONDS", "90"))
        cors = os.getenv(
            "SCENEWORKS_CORS_ORIGINS",
            ",".join(
                [
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://localhost:5174",
                    "http://127.0.0.1:5174",
                    "http://localhost:5175",
                    "http://127.0.0.1:5175",
                    "http://localhost:5176",
                    "http://127.0.0.1:5176",
                ]
            ),
        )
        self.cors_origins = [origin.strip() for origin in cors.split(",") if origin.strip()]

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "recent-projects.json"

    @property
    def jobs_db_path(self) -> Path:
        configured = os.getenv("SCENEWORKS_JOBS_DB_PATH", "").strip()
        if configured:
            return Path(configured).resolve()
        return self.data_dir / "cache" / "jobs.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
