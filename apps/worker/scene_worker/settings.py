import os
import sys
from pathlib import Path


class WorkerSettings:
    def __init__(self) -> None:
        self.worker_id = os.getenv("SCENEWORKS_WORKER_ID", "worker-local-0")
        # Apple Silicon has no NVIDIA card to supervise, so default straight to
        # the MPS backend. Windows/Linux keep "auto" (supervise visible NVIDIA
        # GPUs). An explicit SCENEWORKS_GPU_ID always wins, including "cpu" (sc-1335).
        default_gpu_id = "mps" if sys.platform == "darwin" else "auto"
        self.gpu_id = os.getenv("SCENEWORKS_GPU_ID", default_gpu_id)
        self.api_url = os.getenv("SCENEWORKS_API_URL", "http://localhost:8000")
        self.data_dir = Path(os.getenv("SCENEWORKS_DATA_DIR", "data")).resolve()
        self.config_dir = Path(os.getenv("SCENEWORKS_CONFIG_DIR", "config")).resolve()
        self.access_token = os.getenv("SCENEWORKS_ACCESS_TOKEN", "").strip()
        self.heartbeat_seconds = int(os.getenv("SCENEWORKS_WORKER_HEARTBEAT_SECONDS", "30"))
        self.poll_seconds = int(os.getenv("SCENEWORKS_WORKER_POLL_SECONDS", "3"))
        # Hard-stop backstop: if a running job's cancellation isn't honored
        # cooperatively within this many seconds, the worker force-terminates
        # itself so its supervisor respawns it (releasing the wedged job/model).
        # 0 disables the backstop, leaving cancellation cooperative-only.
        self.force_cancel_seconds = int(os.getenv("SCENEWORKS_WORKER_FORCE_CANCEL_SECONDS", "30"))
        # A GPU worker won't claim a job when its GPU has less free VRAM than this
        # (MB), so jobs flow to a free card when another tool (e.g. ComfyUI) is using
        # one. The 24 GB default is NVIDIA-tuned; at claim time it's clamped to a high
        # fraction of the card's own capacity so a smaller card isn't blocked forever
        # (see should_skip_claim_low_vram). 0 disables the gate. CPU/MPS are never gated.
        self.min_free_vram_mb = int(os.getenv("SCENEWORKS_MIN_FREE_VRAM_MB", "24000"))
        # Prompt refinement (sc-2041) calls an OpenAI-compatible chat-completions
        # endpoint (e.g. a local gemma GGUF served by Ollama/llama.cpp/LM Studio).
        # Left blank by default; when base_url+model are unset the refine job fails
        # fast with a clear "runtime not configured" message instead of sitting queued.
        self.prompt_refine_base_url = os.getenv("PROMPT_REFINE_BASE_URL", "").strip()
        self.prompt_refine_api_key = os.getenv("PROMPT_REFINE_API_KEY", "").strip()
        self.prompt_refine_model = os.getenv("PROMPT_REFINE_MODEL", "").strip()
        self.prompt_refine_timeout_seconds = float(os.getenv("PROMPT_REFINE_TIMEOUT_SECONDS", "60"))
        self.prompt_refine_max_tokens = int(os.getenv("PROMPT_REFINE_MAX_TOKENS", "1024"))

    def for_worker(self, *, worker_id: str, gpu_id: str) -> "WorkerSettings":
        settings = object.__new__(WorkerSettings)
        settings.worker_id = worker_id
        settings.gpu_id = gpu_id
        settings.api_url = self.api_url
        settings.data_dir = self.data_dir
        settings.config_dir = self.config_dir
        settings.access_token = self.access_token
        settings.heartbeat_seconds = self.heartbeat_seconds
        settings.poll_seconds = self.poll_seconds
        settings.min_free_vram_mb = self.min_free_vram_mb
        settings.force_cancel_seconds = self.force_cancel_seconds
        settings.prompt_refine_base_url = self.prompt_refine_base_url
        settings.prompt_refine_api_key = self.prompt_refine_api_key
        settings.prompt_refine_model = self.prompt_refine_model
        settings.prompt_refine_timeout_seconds = self.prompt_refine_timeout_seconds
        settings.prompt_refine_max_tokens = self.prompt_refine_max_tokens
        return settings
