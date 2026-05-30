from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
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
    # Wan A14B MoE LoRAs are saved as a pair (`<name>.high_noise` + `<name>.low_noise`).
    # `path` is the high-noise file (applied to the transformer); `secondary_path`,
    # when set, is the low-noise file applied to the pipeline's transformer_2.
    secondary_path: str | None = None


@dataclass(frozen=True)
class LoraPipelineState:
    key: str = ""
    adapter_names: tuple[str, ...] = ()
    specs: tuple[LoraSpec, ...] = ()


def lora_cache_key(loras: list[dict[str, Any]]) -> str:
    return lora_cache_key_for_specs(normalize_lora_specs(loras))


def lora_cache_key_for_specs(specs: list[LoraSpec]) -> str:
    canonical = [
        {
            "id": spec.id,
            "path": spec.path,
            "weight": spec.weight,
            # Only emit when present so non-MoE keys stay byte-identical to before.
            **({"secondary": spec.secondary_path} if spec.secondary_path else {}),
        }
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
#
# FLUX.2 [klein] is a distilled variant whose model family ("flux2-klein") differs
# from its LoRA-compatible family ("flux2"): klein LoRAs are flux2 LoRAs and are
# detected/declared as such (see lora_family.rs Bucket::Flux2). The klein models'
# manifest loraCompatibility.families is ["flux2"], so a klein model must accept
# flux2 LoRAs even though its own family string is "flux2-klein".
EXTRA_COMPATIBLE_LORA_FAMILIES: dict[str, set[str]] = {
    "chroma": {"flux"},
    "flux2-klein": {"flux2"},
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


def lora_base_model(lora: dict[str, Any]) -> str | None:
    """The specific base model a LoRA was trained for (e.g. wan_2_2,
    wan_2_2_t2v_14b), or None for LoRAs that don't record one."""
    value = lora.get("baseModel") or lora.get("base_model")
    text = str(value).strip() if value is not None else ""
    return text or None


# Families whose members share an architecture family but NOT a LoRA-compatible
# architecture, so a matching family is not enough — the trained base model must
# also match. Wan is the case: wan_2_2 (5B, 48 latent ch) and wan_2_2_*_14b (A14B,
# 16 ch) are both family `wan-video` but cross-applying a LoRA garbles output.
_BASE_MODEL_GATED_FAMILIES = {"wan-video"}


def validate_lora_compatibility(
    loras: list[dict[str, Any]],
    *,
    model_family: str | None,
    adapter_id: str,
    model_id: str | None = None,
) -> None:
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
        # Base-model gating for families where the family alone is insufficient
        # (Wan 5B vs 14B). A LoRA that records its trained base model only applies
        # to that exact model; LoRAs without one fall back to family gating.
        if model_id and not _BASE_MODEL_GATED_FAMILIES.isdisjoint(families):
            base = lora_base_model(lora)
            if base and base != model_id:
                raise RuntimeError(
                    f"LoRA {lora_id} was trained for base model {base}, not {model_id}; "
                    f"Wan 5B and 14B LoRAs are not interchangeable."
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
                secondary_path=wan_moe_low_noise_sibling(path),
            )
        )
    return specs


# Wan A14B MoE trainer (sc-1953) saves a pair: `<stem>.high_noise.safetensors`
# (resolved as the primary, applied to the transformer) and a `.low_noise` sibling
# applied to transformer_2. Match the `.high_noise.` infix so single-file LoRAs
# (5B, image families) are unaffected.
_WAN_MOE_HIGH_NOISE_RE = re.compile(r"\.high_noise\.safetensors$", re.IGNORECASE)


def wan_moe_low_noise_sibling(primary: Path) -> str | None:
    """Return the `.low_noise` sibling of a Wan MoE high-noise LoRA file, or None
    when the resolved file is not the high-noise half of a two-expert pair."""
    if not _WAN_MOE_HIGH_NOISE_RE.search(primary.name):
        return None
    sibling = primary.with_name(_WAN_MOE_HIGH_NOISE_RE.sub(".low_noise.safetensors", primary.name))
    return str(sibling) if sibling.is_file() else None


# --------------------------------------------------------------------------- #
# LoKr (LyCORIS Kronecker) support (epic 2193).
#
# diffusers' ``load_lora_weights`` only understands LoRA-format keys, so a LoKr
# adapter (``lokr_w1``/``lokr_w2``) must be applied by rebuilding its
# ``peft.LoKrConfig`` from the safetensors header and injecting it into the
# pipeline's denoiser module. The trainer (epic 2193 / write_lokr_adapter) stamps
# everything needed: ``networkType`` to route, plus ``rank``/``alpha``/
# ``decomposeFactor``/``targetModules`` to reconstruct the network.
# --------------------------------------------------------------------------- #


def read_adapter_metadata(path: str | Path) -> dict[str, str]:
    """Best-effort read of an adapter's safetensors header metadata."""

    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            return dict(handle.metadata() or {})
    except Exception:
        return {}


def adapter_network_type(path: str | Path) -> str:
    """``"lokr"`` for a LoKr adapter, else ``"lora"`` (the default for every
    adapter without explicit ``networkType`` metadata)."""

    return (read_adapter_metadata(path).get("networkType") or "lora").strip().lower()


def reject_lokr_loras(specs: list[LoraSpec], adapter_id: str) -> None:
    """Guard for backends that cannot apply LoKr (e.g. MLX, whose merge math is
    LoRA-only). Raises a clear error rather than silently mis-applying."""

    for spec in specs:
        if adapter_network_type(spec.path) == "lokr":
            raise RuntimeError(
                f"{adapter_id} cannot apply the LoKr adapter '{spec.id}'. LoKr "
                "(LyCORIS Kronecker) adapters require the torch generation backend; "
                "this backend does not support them yet (epic 2193)."
            )


def _denoiser_module(pipe: Any) -> Any:
    """The pipeline submodule LoKr injects into — the UNet or the DiT transformer."""

    return getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)


def inject_lokr_adapter(pipe: Any, spec: LoraSpec, *, adapter_id: str) -> None:
    """Rebuild ``spec``'s ``LoKrConfig`` from its file metadata and inject it into
    the denoiser, then load the trained weights — the LoKr equivalent of
    ``pipe.load_lora_weights`` (which cannot consume LoKr keys)."""

    module = _denoiser_module(pipe)
    if module is None:
        raise RuntimeError(
            f"{adapter_id} cannot apply the LoKr adapter '{spec.id}': the pipeline "
            "exposes no unet/transformer module to inject into."
        )
    try:
        peft = importlib.import_module("peft")
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(peft_backend_message(adapter_id, [spec])) from exc

    meta = read_adapter_metadata(spec.path)
    rank = int(meta.get("rank") or 16)
    target_modules = json.loads(meta.get("targetModules") or "null")
    config = peft.LoKrConfig(
        r=rank,
        alpha=int(meta.get("alpha") or rank),
        decompose_factor=int(meta.get("decomposeFactor") or -1),
        target_modules=target_modules,
        init_weights=True,
    )
    peft.inject_adapter_in_model(config, module, adapter_name=spec.adapter_name)

    state = load_file(str(spec.path))
    reference = next(module.parameters(), None)
    if reference is not None:
        state = {
            key: value.to(device=reference.device, dtype=reference.dtype)
            for key, value in state.items()
        }
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(module, state, adapter_name=spec.adapter_name)


def apply_loras_to_pipeline(
    pipe: Any,
    loras: list[dict[str, Any]],
    *,
    adapter_id: str,
    model_family: str | None = None,
    model_id: str | None = None,
    previous_state: LoraPipelineState | None = None,
) -> LoraPipelineState:
    previous_state = previous_state or LoraPipelineState()
    validate_lora_compatibility(
        loras, model_family=model_family, adapter_id=adapter_id, model_id=model_id
    )
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
        # LoKr can't load via load_lora_weights (its lokr_* keys are unrecognized);
        # rebuild the LoKrConfig from file metadata and inject it instead (epic 2193).
        if adapter_network_type(spec.path) == "lokr":
            inject_lokr_adapter(pipe, spec, adapter_id=adapter_id)
            continue
        try:
            pipe.load_lora_weights(spec.path, adapter_name=spec.adapter_name)
            # Wan A14B MoE: also load the low-noise half into the second expert.
            # diffusers' WanLoraLoaderMixin routes load_into_transformer_2=True to
            # pipe.transformer_2. Skip if the pipeline has no second expert (a MoE
            # LoRA on a dense model should be blocked upstream by base-model gating).
            if spec.secondary_path and getattr(pipe, "transformer_2", None) is not None:
                pipe.load_lora_weights(
                    spec.secondary_path,
                    adapter_name=spec.adapter_name,
                    load_into_transformer_2=True,
                )
        except Exception as exc:
            if is_peft_backend_error(exc):
                raise RuntimeError(peft_backend_message(adapter_id, [spec])) from exc
            raise

    names = tuple(spec.adapter_name for spec in specs)
    weights = [spec.weight for spec in specs]
    # Injected LoKr adapters live on the denoiser module but aren't tracked by the
    # pipeline's lora bookkeeping, so pipe.set_adapters may not see them. When any
    # adapter is LoKr, set weights on the module (which knows every PEFT adapter).
    if any(adapter_network_type(spec.path) == "lokr" for spec in specs):
        set_adapter_weights_on_module(
            _denoiser_module(pipe), names, weights, adapter_id=adapter_id, specs=specs
        )
    elif hasattr(pipe, "set_adapters"):
        set_lora_adapters(pipe, names, weights, adapter_id=adapter_id, specs=specs)
    elif len(names) > 1 or any(weight != 1.0 for weight in weights):
        raise RuntimeError(f"{adapter_id} loaded LoRAs but cannot apply per-LoRA weights.")
    return LoraPipelineState(key=key, adapter_names=names, specs=tuple(specs))


def clear_loras(pipe: Any, adapter_names: tuple[str, ...], *, adapter_id: str) -> None:
    if not adapter_names:
        return
    # Prefer deleting by name: it removes injected LoKr adapters too, which
    # unload_lora_weights (LoRA-only) leaves behind and would leak into the next
    # job (epic 2193). Fall back to unload for pipelines without delete_adapters.
    if hasattr(pipe, "delete_adapters"):
        pipe.delete_adapters(list(adapter_names))
        return
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()
        return
    raise RuntimeError(f"{adapter_id} cannot clear previously loaded LoRAs between jobs.")


def set_adapter_weights_on_module(
    module: Any,
    names: tuple[str, ...],
    weights: list[float],
    *,
    adapter_id: str,
    specs: list[LoraSpec],
) -> None:
    """Activate adapters and apply per-adapter weights on the denoiser module
    itself — used when LoKr adapters are present (see ``apply_loras_to_pipeline``)."""

    if module is None or not hasattr(module, "set_adapters"):
        # A single adapter at full weight needs no explicit activation; anything
        # else genuinely cannot be applied without module-level adapter control.
        if len(names) > 1 or any(weight != 1.0 for weight in weights):
            raise RuntimeError(
                f"{adapter_id} loaded a LoKr adapter but cannot apply per-adapter weights."
            )
        return
    try:
        module.set_adapters(list(names), weights=list(weights))
    except Exception as exc:
        if is_peft_backend_error(exc):
            raise RuntimeError(peft_backend_message(adapter_id, specs)) from exc
        raise


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
