from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .hf_cache import huggingface_repo_cache_path


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


# Architecture families a model can load LoRAs from in addition to its own.
# Chroma is FLUX.1-schnell-derived and keeps Flux's transformer-block layout, so
# Flux LoRAs load on Chroma — and Chroma LoRAs that carry no chroma metadata are
# classified as `flux` by the key-based detector (the keys are identical). The
# relationship is one-directional: a Flux model does not accept chroma LoRAs.
EXTRA_COMPATIBLE_LORA_FAMILIES: dict[str, set[str]] = {
    "chroma": {"flux"},
}


def accepted_lora_families(model_family: Any) -> set[str]:
    """The set of LoRA families a model of ``model_family`` can load."""
    normalized = normalize_lora_family(model_family)
    if not normalized:
        return set()
    return {normalized} | EXTRA_COMPATIBLE_LORA_FAMILIES.get(normalized, set())


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
    if lora.get("icLora") is True or lora.get("isIcLora") is True:
        return True
    if str(lora.get("conditioningRole") or "").strip().lower().replace("-", "_") == "ic_lora":
        return True
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
    accepted = accepted_lora_families(model_family)
    if not loras or not accepted:
        return
    for index, item in enumerate(loras):
        lora = item if isinstance(item, dict) else {"id": str(item)}
        lora_id = str(lora.get("id") or lora.get("loraId") or f"lora_{index + 1}").strip()
        families = lora_families(lora)
        if not families:
            continue
        # Accept when any of the LoRA's declared families is one the model can
        # load. For a single-family model this is exactly "model_family in families";
        # chroma additionally accepts flux (see EXTRA_COMPATIBLE_LORA_FAMILIES).
        if accepted.isdisjoint(families):
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
    if not value:
        return huggingface_cached_lora_path(lora)
    path = Path(str(value)).expanduser()
    if path.exists():
        return resolve_lora_file(path, lora)
    return huggingface_cached_lora_path(lora) or path


def resolve_lora_file(path: Path, lora: dict[str, Any]) -> Path:
    # Installed LoRAs are stored as a directory (the manifest keeps the file name
    # in `files`; `source.path`/`installedPath` point at the directory). The native
    # ltx-core loader mmaps the given path directly, and mmap on a directory fails
    # with ENODEV ("No such device (os error 19)"), so descend to the actual
    # .safetensors file. Diffusers accepts a file too, so this is safe for every
    # adapter.
    if not path.is_dir():
        return path
    for name in lora_declared_files(lora):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return first_safetensors_path(path) or path


def lora_declared_files(lora: dict[str, Any]) -> list[str]:
    source = lora.get("source") if isinstance(lora.get("source"), dict) else {}
    raw = lora.get("files") or source.get("files") or source.get("file") or lora.get("file") or []
    files = raw if isinstance(raw, list) else [raw]
    return [str(name).strip() for name in files if str(name).strip()]


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
    main_snapshot = huggingface_main_snapshot_dir(root)
    if main_snapshot is not None:
        safetensors_path = first_safetensors_path(main_snapshot)
        if safetensors_path is not None:
            return safetensors_path
    return first_safetensors_path(root)


def huggingface_snapshot_dirs(repo_root: Path) -> list[Path]:
    snapshots = repo_root / "snapshots"
    if not snapshots.is_dir():
        return []
    candidates = sorted(candidate for candidate in snapshots.iterdir() if candidate.is_dir())
    main_snapshot = huggingface_main_snapshot_dir(repo_root)
    if main_snapshot is None:
        return candidates
    return [main_snapshot, *[candidate for candidate in candidates if candidate != main_snapshot]]


def huggingface_main_snapshot_dir(repo_root: Path) -> Path | None:
    ref_path = repo_root / "refs" / "main"
    try:
        revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not revision:
        return None
    snapshot = repo_root / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def path_stem(path: Path | None) -> str | None:
    return path.stem if path else None


# SceneWorks training writes per-step checkpoints (`<stem>-step000250.safetensors`)
# next to the final adapter. Recognize them so the fallback never auto-selects a
# checkpoint over the final weights.
_LORA_CHECKPOINT_RE = re.compile(r"-step\d{6}\.safetensors$", re.IGNORECASE)


def first_safetensors_path(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".safetensors":
        return path
    if not path.is_dir():
        return None
    candidates = sorted(
        (candidate for candidate in path.rglob("*.safetensors") if candidate.is_file()),
        key=lambda candidate: candidate.as_posix(),
    )
    if not candidates:
        return None
    # Prefer a non-checkpoint adapter; the name-sorted first file is the *earliest*
    # (least-trained) checkpoint, which is exactly the wrong one to load. If only
    # checkpoints exist, fall back to the highest step (most-trained).
    finals = [candidate for candidate in candidates if not _LORA_CHECKPOINT_RE.search(candidate.name)]
    if finals:
        return finals[0]
    return max(candidates, key=lambda candidate: candidate.name)


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
