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
        self.hf_home = Path(os.getenv("SCENEWORKS_HF_HOME") or os.getenv("HF_HOME") or self.data_dir / "cache" / "huggingface").resolve()
        self.access_token = os.getenv("SCENEWORKS_ACCESS_TOKEN", "").strip()
        self.huggingface_token = (
            os.getenv("SCENEWORKS_HF_TOKEN", "").strip()
            or os.getenv("HF_TOKEN", "").strip()
            or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
        )
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
        return self.data_dir / "jobs.db"

    @property
    def manifests_dir(self) -> Path:
        return self.config_dir / "manifests"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def loras_dir(self) -> Path:
        return self.data_dir / "loras"

    @property
    def hf_cache_dir(self) -> Path:
        return Path(os.getenv("HF_HUB_CACHE") or self.hf_home / "hub").resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
