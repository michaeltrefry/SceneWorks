from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
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


def normalize_lora_family(family: Any) -> str:
    return str(family or "").strip().lower().replace("_", "-")


def lora_families(lora: dict[str, Any]) -> list[str]:
    compatibility = lora.get("compatibility") if isinstance(lora.get("compatibility"), dict) else {}
    values = next(
        (
            candidate
            for candidate in (
                lora.get("families"),
                lora.get("compatibleFamilies"),
                lora.get("modelFamilies"),
                compatibility.get("families"),
                [lora.get("family")] if lora.get("family") else None,
            )
            if candidate is not None
        ),
        [],
    )
    if not isinstance(values, list):
        values = [values]
    families = sorted({normalize_lora_family(value) for value in values if normalize_lora_family(value)})
    return families


def lora_looks_like_ic_lora(lora: dict[str, Any]) -> bool:
    source = lora.get("source") if isinstance(lora.get("source"), dict) else {}
    files = source.get("files") or lora.get("files") or []
    if not isinstance(files, list):
        files = [files]
    values = [
        lora.get("id"),
        lora.get("loraId"),
        lora.get("name"),
        lora.get("displayName"),
        lora.get("installedPath"),
        lora.get("sourcePath"),
        lora.get("path"),
        source.get("repo"),
        source.get("file"),
        source.get("path"),
        *files,
    ]
    text = " ".join(str(value) for value in values if value).lower().replace("_", "-")
    return "ic-lora" in text or "ltx-2-3-ic-" in text


def validate_lora_compatibility(loras: list[dict[str, Any]], *, model_family: str | None, adapter_id: str) -> None:
    normalized_model_family = normalize_lora_family(model_family)
    if not loras or not normalized_model_family:
        return
    for index, item in enumerate(loras):
        lora = item if isinstance(item, dict) else {"id": str(item)}
        lora_id = str(lora.get("id") or lora.get("loraId") or f"lora_{index + 1}").strip()
        families = lora_families(lora)
        if not families:
            continue
        if normalized_model_family not in families:
            raise RuntimeError(
                f"LoRA {lora_id} is not compatible with model family {normalized_model_family} for {adapter_id}."
            )


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
        if path.is_dir() and not first_safetensors_path(path):
            raise RuntimeError(f"LoRA {lora_id} has no .safetensors file under: {path}")
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
    model_family: str | None = None,
    previous_state: LoraPipelineState | None = None,
) -> LoraPipelineState:
    previous_state = previous_state or LoraPipelineState()
    validate_lora_compatibility(loras, model_family=model_family, adapter_id=adapter_id)
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
        try:
            pipe.load_lora_weights(spec.path, adapter_name=spec.adapter_name)
        except Exception as exc:
            if is_peft_backend_error(exc):
                raise RuntimeError(peft_backend_message(adapter_id, [spec])) from exc
            raise

    names = tuple(spec.adapter_name for spec in specs)
    weights = [spec.weight for spec in specs]
    if hasattr(pipe, "set_adapters"):
        set_lora_adapters(pipe, names, weights, adapter_id=adapter_id, specs=specs)
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
    fallback = huggingface_cached_lora_path(lora)
    if not value:
        return fallback
    path = Path(str(value)).expanduser()
    if path.exists() or fallback is None:
        return path
    return fallback


def huggingface_cached_lora_path(lora: dict[str, Any]) -> Path | None:
    source = lora.get("source") if isinstance(lora.get("source"), dict) else {}
    provider = str(source.get("provider") or lora.get("provider") or "").strip().lower()
    if provider != "huggingface":
        return None
    repo = str(source.get("repo") or lora.get("repo") or "").strip()
    if not repo:
        return None
    root = huggingface_repo_cache_path(repo)
    if root is None or not root.exists():
        return None
    file_name = source.get("file") or lora.get("file")
    if not file_name:
        files = source.get("files") or lora.get("files")
        if isinstance(files, list) and files:
            file_name = files[0]
    if file_name:
        for snapshot in huggingface_snapshot_dirs(root):
            candidate = snapshot / str(file_name)
            if candidate.is_file():
                return candidate
    return first_safetensors_path(root)


def huggingface_repo_cache_path(repo: str) -> Path | None:
    safe_repo = "".join(character if character.isalnum() or character in "._-" else "--" for character in repo).strip("-")
    if not safe_repo:
        return None
    return huggingface_hub_cache_dir() / f"models--{safe_repo}"


def huggingface_hub_cache_dir() -> Path:
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(key, "").strip()
        if value:
            return Path(value).expanduser()
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    data_dir = Path(os.getenv("SCENEWORKS_DATA_DIR", "data")).expanduser()
    return data_dir / "cache" / "huggingface" / "hub"


def huggingface_snapshot_dirs(repo_root: Path) -> list[Path]:
    snapshots = repo_root / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(candidate for candidate in snapshots.iterdir() if candidate.is_dir())


def path_stem(path: Path | None) -> str | None:
    return path.stem if path else None


def first_safetensors_path(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".safetensors":
        return path
    if not path.is_dir():
        return None
    return next((candidate for candidate in path.rglob("*.safetensors") if candidate.is_file()), None)


def lora_weight(lora: dict[str, Any]) -> float:
    try:
        return float(lora.get("weight", lora.get("defaultWeight", 0.8)))
    except (TypeError, ValueError):
        return 0.8


def set_lora_adapters(pipe: Any, names: tuple[str, ...], weights: list[float], *, adapter_id: str, specs: list[LoraSpec]) -> None:
    try:
        # Newer Diffusers releases use adapter_weights; older builds used weights.
        pipe.set_adapters(list(names), adapter_weights=weights)
        return
    except TypeError as exc:
        if is_peft_backend_error(exc):
            raise RuntimeError(peft_backend_message(adapter_id, specs)) from exc

    try:
        pipe.set_adapters(list(names), weights=weights)
    except Exception as exc:
        if is_peft_backend_error(exc):
            raise RuntimeError(peft_backend_message(adapter_id, specs)) from exc
        raise


def is_peft_backend_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    peft_markers = (
        "peft backend",
        "requires peft",
        "peft is required",
        "install peft",
        "no module named 'peft'",
        'no module named "peft"',
    )
    return (
        isinstance(exc, (ImportError, ModuleNotFoundError)) and "peft" in lowered
    ) or any(marker in lowered for marker in peft_markers)


def peft_backend_message(adapter_id: str, specs: list[LoraSpec]) -> str:
    lora_ids = ", ".join(spec.id for spec in specs) or "selected LoRA"
    return (
        f"LoRA {lora_ids} requires the PEFT backend for {adapter_id}. "
        "For bare-metal workers, run `pip install -r apps/worker/requirements.txt`; "
        "for Docker Compose, run `docker compose build worker --no-cache`, then restart the worker and retry the preset."
    )


def safe_adapter_name(lora_id: str, path: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", lora_id).strip("_") or "lora"
    digest = hashlib.sha256(f"{lora_id}:{path}".encode("utf-8")).hexdigest()[:10]
    return f"sw_{safe_id[:40]}_{digest}"
