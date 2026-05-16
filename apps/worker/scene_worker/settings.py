import os
from pathlib import Path


class WorkerSettings:
    def __init__(self) -> None:
        self.worker_id = os.getenv("SCENEWORKS_WORKER_ID", "worker-local-0")
        self.gpu_id = os.getenv("SCENEWORKS_GPU_ID", "auto")
        self.api_url = os.getenv("SCENEWORKS_API_URL", "http://localhost:8000")
        self.data_dir = Path(os.getenv("SCENEWORKS_DATA_DIR", "data")).resolve()
        self.config_dir = Path(os.getenv("SCENEWORKS_CONFIG_DIR", "config")).resolve()
        self.access_token = os.getenv("SCENEWORKS_ACCESS_TOKEN", "").strip()
        self.heartbeat_seconds = int(os.getenv("SCENEWORKS_WORKER_HEARTBEAT_SECONDS", "30"))
