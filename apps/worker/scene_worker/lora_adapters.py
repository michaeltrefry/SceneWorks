from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any


# Keep in sync with packages/schemas/recipe-preset.schema.json and the Rust
# normalize_recipe_preset_loras API guard.
MAX_JOB_LORAS = 3


@dataclass(frozen=True)
class LoraSpec:
    id: str
    path: str
    weight: float
    adapter_name: str


@dataclass(frozen=True)
class LoraPipelineState:
    key: str = ""
    adapter_names: tuple[str, ...] = ()
    specs: tuple[LoraSpec, ...] = ()


def lora_cache_key(loras: list[dict[str, Any]]) -> str:
    return lora_cache_key_for_specs(normalize_lora_specs(loras))


def lora_cache_key_for_specs(specs: list[LoraSpec]) -> str:
    canonical = [
        {"id": spec.id, "path": spec.path, "weight": spec.weight}
        for spec in sorted(specs, key=lambda item: (item.id, item.path, item.weight))
    ]
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reject_loras_if_unsupported(loras: list[dict[str, Any]], adapter_id: str) -> None:
    if loras:
        raise RuntimeError(f"{adapter_id} does not support LoRA application for generation jobs.")


def normalize_lora_specs(loras: list[dict[str, Any]]) -> list[LoraSpec]:
    if len(loras) > MAX_JOB_LORAS:
        raise RuntimeError(f"Generation supports at most {MAX_JOB_LORAS} LoRAs per job.")

    specs = []
    for index, item in enumerate(loras):
        lora = item if isinstance(item, dict) else {"id": str(item)}
        path = lora_path(lora)
        lora_id = str(lora.get("id") or lora.get("loraId") or path_stem(path) or f"lora_{index + 1}").strip()
        if path is None:
            raise RuntimeError(f"LoRA {lora_id} is not installed. Import or download it before generation.")
        if not path.exists():
            raise RuntimeError(f"LoRA {lora_id} file is missing: {path}")
        path_text = str(path)
        specs.append(
            LoraSpec(
                id=lora_id,
                path=path_text,
                weight=lora_weight(lora),
                adapter_name=safe_adapter_name(lora_id, path_text),
            )
        )
    return specs


def apply_loras_to_pipeline(
    pipe: Any,
    loras: list[dict[str, Any]],
    *,
    adapter_id: str,
    previous_state: LoraPipelineState | None = None,
) -> LoraPipelineState:
    previous_state = previous_state or LoraPipelineState()
    specs = normalize_lora_specs(loras)
    key = lora_cache_key_for_specs(specs)
    if key == previous_state.key:
        return previous_state

    if not specs:
        clear_loras(pipe, previous_state.adapter_names, adapter_id=adapter_id)
        return LoraPipelineState()
    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError(f"{adapter_id} does not support loading LoRA weights.")

    previous_by_name = {spec.adapter_name: spec for spec in previous_state.specs}
    desired_by_name = {spec.adapter_name: spec for spec in specs}
    removed_names = tuple(name for name in previous_state.adapter_names if name not in desired_by_name)
    specs_to_load = [spec for spec in specs if spec.adapter_name not in previous_by_name]

    if removed_names:
        if hasattr(pipe, "delete_adapters"):
            pipe.delete_adapters(list(removed_names))
        elif hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
            specs_to_load = specs
        else:
            raise RuntimeError(f"{adapter_id} cannot clear previously loaded LoRAs between jobs.")

    for spec in specs_to_load:
        pipe.load_lora_weights(spec.path, adapter_name=spec.adapter_name)

    names = tuple(spec.adapter_name for spec in specs)
    weights = [spec.weight for spec in specs]
    if hasattr(pipe, "set_adapters"):
        try:
            # Newer Diffusers releases use adapter_weights; older builds used weights.
            pipe.set_adapters(list(names), adapter_weights=weights)
        except TypeError:
            pipe.set_adapters(list(names), weights=weights)
    elif len(names) > 1 or any(weight != 1.0 for weight in weights):
        raise RuntimeError(f"{adapter_id} loaded LoRAs but cannot apply per-LoRA weights.")
    return LoraPipelineState(key=key, adapter_names=names, specs=tuple(specs))


def clear_loras(pipe: Any, adapter_names: tuple[str, ...], *, adapter_id: str) -> None:
    if not adapter_names:
        return
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()
        return
    if hasattr(pipe, "delete_adapters"):
        pipe.delete_adapters(list(adapter_names))
        return
    raise RuntimeError(f"{adapter_id} cannot clear previously loaded LoRAs between jobs.")


def lora_path(lora: dict[str, Any]) -> Path | None:
    source = lora.get("source") if isinstance(lora.get("source"), dict) else {}
    value = lora.get("installedPath") or lora.get("sourcePath") or lora.get("path") or source.get("path")
    if not value:
        return None
    return Path(str(value)).expanduser()


def path_stem(path: Path | None) -> str | None:
    return path.stem if path else None


def lora_weight(lora: dict[str, Any]) -> float:
    try:
        return float(lora.get("weight", lora.get("defaultWeight", 0.8)))
    except (TypeError, ValueError):
        return 0.8


def safe_adapter_name(lora_id: str, path: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", lora_id).strip("_") or "lora"
    digest = hashlib.sha256(f"{lora_id}:{path}".encode("utf-8")).hexdigest()[:10]
    return f"sw_{safe_id[:40]}_{digest}"
