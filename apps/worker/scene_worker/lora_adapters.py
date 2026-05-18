from __future__ import annotations

import re
from pathlib import Path
from typing import Any


MAX_JOB_LORAS = 3


def lora_cache_key(loras: list[dict[str, Any]]) -> str:
    specs = normalize_lora_specs(loras)
    return "|".join(f"{spec['id']}@{spec['path']}#{spec['weight']:g}" for spec in specs)


def assert_loras_supported(loras: list[dict[str, Any]], adapter_id: str) -> None:
    if loras:
        raise RuntimeError(f"{adapter_id} does not support LoRA application for generation jobs.")


def normalize_lora_specs(loras: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(loras) > MAX_JOB_LORAS:
        raise RuntimeError(f"Generation supports at most {MAX_JOB_LORAS} LoRAs per job.")

    specs = []
    for index, item in enumerate(loras):
        lora = item if isinstance(item, dict) else {"id": str(item)}
        lora_id = str(lora.get("id") or lora.get("loraId") or f"lora_{index + 1}").strip()
        path = lora_path(lora)
        if path is None:
            raise RuntimeError(f"LoRA {lora_id} is not installed. Import or download it before generation.")
        if not path.exists():
            raise RuntimeError(f"LoRA {lora_id} file is missing: {path}")
        specs.append(
            {
                "id": lora_id,
                "path": str(path),
                "weight": lora_weight(lora),
                "adapterName": safe_adapter_name(lora_id, index),
            }
        )
    return specs


def apply_loras_to_pipeline(
    pipe: Any,
    loras: list[dict[str, Any]],
    *,
    adapter_id: str,
    previous_key: str = "",
    previous_adapter_names: list[str] | None = None,
) -> tuple[str, list[str]]:
    specs = normalize_lora_specs(loras)
    key = lora_cache_key(loras)
    if key == previous_key:
        return previous_key, previous_adapter_names or []

    clear_loras(pipe, previous_adapter_names or [], adapter_id=adapter_id)
    if not specs:
        return "", []
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{adapter_id} does not support loading LoRA weights.")

    names = []
    for spec in specs:
        pipe.load_lora_weights(spec["path"], adapter_name=spec["adapterName"])
        names.append(spec["adapterName"])

    weights = [spec["weight"] for spec in specs]
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except TypeError:
            pipe.set_adapters(names, weights=weights)
    elif len(names) > 1 or any(weight != 1.0 for weight in weights):
        raise RuntimeError(f"{adapter_id} loaded LoRAs but cannot apply per-LoRA weights.")
    return key, names


def clear_loras(pipe: Any, adapter_names: list[str], *, adapter_id: str) -> None:
    if not adapter_names:
        return
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()
        return
    if hasattr(pipe, "delete_adapters"):
        pipe.delete_adapters(adapter_names)
        return
    raise RuntimeError(f"{adapter_id} cannot clear previously loaded LoRAs between jobs.")


def lora_path(lora: dict[str, Any]) -> Path | None:
    value = (
        lora.get("installedPath")
        or lora.get("sourcePath")
        or lora.get("path")
        or (lora.get("source") if isinstance(lora.get("source"), str) else None)
        or (lora.get("source") or {}).get("path")
    )
    if not value:
        return None
    return Path(str(value)).expanduser()


def lora_weight(lora: dict[str, Any]) -> float:
    try:
        return float(lora.get("weight", lora.get("defaultWeight", 0.8)))
    except (TypeError, ValueError):
        return 0.8


def safe_adapter_name(lora_id: str, index: int) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", lora_id).strip("_") or "lora"
    return f"sw_{index + 1}_{safe_id[:48]}"
