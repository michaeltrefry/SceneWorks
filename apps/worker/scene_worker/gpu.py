from __future__ import annotations

import os
import subprocess


def discover_gpu(requested_gpu_id: str) -> dict:
    if requested_gpu_id and requested_gpu_id != "auto":
        return {
            "id": requested_gpu_id,
            "name": f"GPU {requested_gpu_id}",
            "capabilities": ["placeholder", "gpu"],
        }

    visible_devices = os.getenv("NVIDIA_VISIBLE_DEVICES", "").strip()
    if visible_devices and visible_devices not in ("all", "void", "none"):
        gpu_id = visible_devices.split(",")[0].strip()
        return {"id": gpu_id, "name": f"GPU {gpu_id}", "capabilities": ["placeholder", "gpu"]}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        first_line = result.stdout.strip().splitlines()[0]
        index, name, memory_mb = [part.strip() for part in first_line.split(",", maxsplit=2)]
        return {
            "id": index,
            "name": f"{name} ({memory_mb} MB)",
            "capabilities": ["placeholder", "gpu", "nvidia"],
        }
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return {
            "id": "cpu",
            "name": "CPU placeholder",
            "capabilities": ["placeholder", "cpu"],
        }
