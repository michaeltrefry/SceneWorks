"""Flow-compatible sampler / scheduler registry (epic 1753).

Swaps a diffusers pipeline's ``pipe.scheduler`` between flow-matching solvers
(Euler, Heun, DPM++ flow mode, UniPC flow mode) and sigma-schedule variants
(simple / shift / karras / exponential / beta). The selection rides the
job's ``advanced`` dict end-to-end — Rust/TS contracts don't change.

Design notes:
- ``apply_sampler(pipe, sampler, scheduler, shift)`` is the single entry
  point. Both ``"default"`` selections are a true no-op (the pipe's loaded
  scheduler is left untouched).
- New schedulers are built via
  ``Cls.from_config(pipe.scheduler.config, **overrides)`` so the model's
  trained schedule params (num_train_timesteps, shift base, etc.) carry
  over and only the chosen axis is overridden.
- Unknown sampler / scheduler names fall back to ``"default"`` and emit a
  worker event; we never hard-fail a generation over a sampling knob.
- Diffusers version drift: ``use_beta_sigmas`` / ``use_exponential_sigmas``
  / ``use_karras_sigmas`` exist only on some scheduler classes and only in
  recent diffusers builds. The registry feature-detects the destination
  scheduler's accepted config keys and drops unsupported flags instead of
  raising.
- The original scheduler (class + config) is snapshotted on first observed
  apply so a later ``("default", "default")`` request can restore it. The
  snapshot rides on the pipe instance via a private attribute — survives
  pipeline cache hits, evicted with the pipeline.
- Path-agnostic: callable from any adapter that loads a diffusers pipe
  (image: z-image, qwen, FLUX-flow models; video: Wan torch). Non-flow-
  matching pipes simply opt out by not calling this function.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    """Self-contained UTC ISO-8601 timestamp.

    Mirrors ``sceneworks_shared.utc_now`` so the registry can also load inside
    the Lens / Lens-Turbo sidecar venv, which does not ship the shared package.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


SUPPORTED_SAMPLERS: tuple[str, ...] = (
    "default",
    "euler",
    "euler_a",
    "heun",
    "dpmpp",
    "dpmpp_sde",
    "unipc",
)
SUPPORTED_SCHEDULERS: tuple[str, ...] = (
    "default",
    "simple",
    "shift",
    "karras",
    "exponential",
    "beta",
)


@dataclass(frozen=True)
class SamplerSpec:
    """Map a sampler key onto a diffusers scheduler class + base overrides."""

    scheduler_cls: str
    config_overrides: dict[str, Any]


# Flow-matching scheduler specs — used for flow pipelines (Z-Image, Qwen, FLUX,
# Wan, …). UNCHANGED from epic 1753; the family selector below routes flow pipes
# here so their behaviour is byte-identical.
_FLOW_SAMPLERS: dict[str, SamplerSpec | None] = {
    "default": None,
    "euler": SamplerSpec("FlowMatchEulerDiscreteScheduler", {}),
    "heun": SamplerSpec("FlowMatchHeunDiscreteScheduler", {}),
    # ``use_dynamic_shifting=True`` is what lets a non-FlowMatch scheduler
    # accept the ``mu`` kwarg that flow-pipeline ``__call__`` methods (Z-Image,
    # Qwen, FLUX, …) unconditionally forward via diffusers ``retrieve_timesteps``.
    # Without it DPMSolver's ``set_timesteps`` asserts on ``mu is not None``;
    # UniPC silently discards ``mu`` (i.e. the trained dynamic shift never lands).
    # Pinning the flag here makes both classes honor the pipeline-computed mu.
    "dpmpp": SamplerSpec(
        "DPMSolverMultistepScheduler",
        {
            "use_flow_sigmas": True,
            "prediction_type": "flow_prediction",
            "use_dynamic_shifting": True,
        },
    ),
    "unipc": SamplerSpec(
        "UniPCMultistepScheduler",
        {"use_flow_sigmas": True, "use_dynamic_shifting": True},
    ),
}

# Standard (epsilon / v-prediction) scheduler specs — used for SDXL-family pipes
# (InstantID/RealVisXL, Kolors, base SDXL). These are the classic diffusers
# solvers WITHOUT the flow flags; pairing them with the ``karras`` scheduler axis
# gives e.g. "DPM++ 2M Karras" (``dpmpp``) and "DPM++ SDE Karras" (``dpmpp_sde``,
# the RealVisXL-recommended combo). ``dpmpp_sde`` uses the multistep
# ``sde-dpmsolver++`` algorithm so no extra ``torchsde`` dependency is needed.
_STD_SAMPLERS: dict[str, SamplerSpec | None] = {
    "default": None,
    "euler": SamplerSpec("EulerDiscreteScheduler", {}),
    "euler_a": SamplerSpec("EulerAncestralDiscreteScheduler", {}),
    "heun": SamplerSpec("HeunDiscreteScheduler", {}),
    "dpmpp": SamplerSpec("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++"}),
    "dpmpp_sde": SamplerSpec("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"}),
    "unipc": SamplerSpec("UniPCMultistepScheduler", {}),
}

# Per-scheduler sigma-spacing flags. "simple" explicitly turns the alternates
# off so a sticky-config from a prior run can't bleed through.
_SCHEDULER_SIGMA_FLAGS: dict[str, dict[str, bool]] = {
    "default": {},
    "simple": {
        "use_karras_sigmas": False,
        "use_exponential_sigmas": False,
        "use_beta_sigmas": False,
    },
    "shift": {},
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
    "beta": {"use_beta_sigmas": True},
}


def _emit(event: str, **payload: Any) -> None:
    payload["event"] = event
    payload["reportedAt"] = _utc_now()
    try:
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        sys.stdout.flush()
    except Exception:
        # The sampler swap must never bring a generation down.
        pass


def _scheduler_accepted_params(cls: Any) -> set[str]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set()
    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    params = set(sig.parameters)
    if has_var_keyword:
        # __init__ accepts **kwargs — diffusers will route unknowns through
        # the config system. Don't pre-filter.
        return params | {"*"}
    return params


def _filter_overrides(cls: Any, overrides: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop overrides the scheduler class does not accept."""
    accepted = _scheduler_accepted_params(cls)
    if "*" in accepted:
        return dict(overrides), []
    kept = {key: value for key, value in overrides.items() if key in accepted}
    dropped = [key for key in overrides if key not in accepted]
    return kept, dropped


def _coerce_shift(shift: Any) -> float | None:
    if shift is None:
        return None
    try:
        value = float(shift)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return value


def _normalize(name: Any, table: tuple[str, ...]) -> str:
    if not isinstance(name, str):
        return "default"
    key = name.strip().lower()
    return key if key in table else "default"


def _snapshot_original(pipe: Any) -> None:
    if pipe is None or not hasattr(pipe, "scheduler"):
        return
    if getattr(pipe, "_sceneworks_original_scheduler", None) is not None:
        return
    scheduler = pipe.scheduler
    if scheduler is None:
        return
    config = getattr(scheduler, "config", None)
    snapshot_config = _coerce_config(config)
    pipe._sceneworks_original_scheduler = {
        "cls": type(scheduler),
        "config": snapshot_config,
    }


def _scheduler_family(pipe: Any) -> str:
    """Classify the pipe's NATIVE scheduler as ``"flow"`` or ``"standard"``.

    Flow-matching pipes (Z-Image, Qwen, FLUX, Wan) load a ``FlowMatch*``
    scheduler / ``flow_prediction`` config; SDXL-family pipes (InstantID,
    Kolors, base SDXL) load epsilon/v-prediction solvers. The selection drives
    which spec table ``apply_sampler`` uses so an SDXL pipe never gets a
    flow-mode scheduler (which would break its dynamics) and a flow pipe stays
    byte-identical to the epic-1753 behaviour.

    Detection reads the SNAPSHOT of the original scheduler when present (so a
    prior swap on a cached pipe can't flip the classification), else the live
    scheduler.
    """
    snapshot = getattr(pipe, "_sceneworks_original_scheduler", None)
    if snapshot:
        name = getattr(snapshot.get("cls"), "__name__", "") or ""
        config = snapshot.get("config") or {}
    else:
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None:
            return "standard"
        name = type(scheduler).__name__
        config = _coerce_config(getattr(scheduler, "config", None))
    if name.startswith("FlowMatch"):
        return "flow"
    prediction = config.get("prediction_type") if hasattr(config, "get") else None
    if isinstance(prediction, str) and "flow" in prediction:
        return "flow"
    if hasattr(config, "get") and config.get("use_flow_sigmas"):
        return "flow"
    return "standard"


def _coerce_config(config: Any) -> dict[str, Any]:
    """Return a plain-dict view of a diffusers FrozenDict / SimpleNamespace /
    dict-shaped config. Diffusers itself ships a FrozenDict that supports
    ``dict(config)`` and ``.items()``; the worker may also see namespaced
    configs in tests."""
    if config is None:
        return {}
    if hasattr(config, "items"):
        try:
            return dict(config.items())
        except Exception:  # noqa: BLE001
            return {}
    if hasattr(config, "__dict__"):
        return {key: value for key, value in vars(config).items() if not key.startswith("_")}
    try:
        return dict(config)
    except (TypeError, ValueError):
        return {}


def _set_timesteps_accepts_mu(scheduler: Any) -> bool:
    """Best-effort check: does ``scheduler.set_timesteps`` accept a ``mu`` kwarg?

    Used to decide whether to install a mu-absorbing shim
    (``_install_mu_shim``). Returns ``True`` when ``mu`` is a named param OR
    the method accepts ``**kwargs``; ``False`` only when both are absent.
    A signature inspection failure conservatively returns ``True`` so we
    don't shim something we can't reason about.
    """
    set_timesteps = getattr(scheduler, "set_timesteps", None)
    if set_timesteps is None:
        return True
    try:
        sig = inspect.signature(set_timesteps)
    except (TypeError, ValueError):
        return True
    params = sig.parameters
    if "mu" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _install_mu_shim(scheduler: Any, adapter: str | None) -> bool:
    """Wrap ``scheduler.set_timesteps`` so callers passing ``mu=…`` don't crash.

    Required for ``FlowMatchHeunDiscreteScheduler`` whose ``set_timesteps``
    signature is ``(num_inference_steps, device=None)`` — no ``mu``, no
    ``**kwargs``. The flow pipelines (Z-Image, Qwen, FLUX, …) always forward
    ``mu`` through ``retrieve_timesteps``, so a plain swap into Heun raises
    ``TypeError: ... unexpected keyword argument 'mu'``.

    The shim translates ``mu`` to the static-shift form Heun *does* support:
    ``scheduler.config.shift = exp(mu)`` (the same translation DPMSolver does
    internally when ``use_dynamic_shifting`` is on), then delegates to the
    real ``set_timesteps`` without the ``mu`` kwarg. Returns ``True`` when
    the shim was installed.
    """
    set_timesteps = getattr(scheduler, "set_timesteps", None)
    if set_timesteps is None:
        return False
    if getattr(scheduler, "_sceneworks_mu_shim", False):
        return False

    def shim(*args: Any, **kwargs: Any) -> Any:
        mu_value = kwargs.pop("mu", None)
        if mu_value is not None:
            try:
                shift_value = float(math.exp(float(mu_value)))
            except (TypeError, ValueError):
                shift_value = None
            if shift_value is not None:
                config = getattr(scheduler, "config", None)
                if config is not None:
                    try:
                        # Diffusers FrozenDict supports item assignment; both
                        # FrozenDict and SimpleNamespace accept attribute set.
                        if hasattr(config, "__setitem__"):
                            try:
                                config["shift"] = shift_value
                            except (TypeError, KeyError):
                                pass
                        if hasattr(config, "shift"):
                            try:
                                config.shift = shift_value
                            except (AttributeError, TypeError):
                                pass
                    except Exception:  # noqa: BLE001
                        pass
        return set_timesteps(*args, **kwargs)

    scheduler.set_timesteps = shim
    scheduler._sceneworks_mu_shim = True
    _emit(
        "sampler_mu_shim_installed",
        adapter=adapter,
        schedulerClass=type(scheduler).__name__,
    )
    return True


def _restore_original(pipe: Any) -> bool:
    """Re-build the snapshotted scheduler. Returns True only when the pipe
    had actually been swapped — a never-mutated pipe stays untouched."""
    if not getattr(pipe, "_sceneworks_sampler_dirty", False):
        return False
    snapshot = getattr(pipe, "_sceneworks_original_scheduler", None)
    if not snapshot:
        return False
    cls = snapshot.get("cls")
    config = snapshot.get("config") or {}
    if cls is None:
        return False
    try:
        pipe.scheduler = cls.from_config(config)
    except Exception:  # noqa: BLE001
        return False
    pipe._sceneworks_sampler_dirty = False
    return True


def apply_sampler(
    pipe: Any,
    sampler: Any,
    scheduler: Any,
    shift: Any = None,
    *,
    adapter: str | None = None,
) -> dict[str, Any]:
    """Swap ``pipe.scheduler`` based on the requested sampler / scheduler.

    Returns a small dict describing the applied selection — useful for the
    adapter to thread into its `rawAdapterSettings` for reproducibility or
    log. Both axes default to model-native (no-op) when omitted or unknown.
    """

    if pipe is None or not hasattr(pipe, "scheduler"):
        return {"sampler": "default", "scheduler": "default", "noop": True}

    _snapshot_original(pipe)

    sampler_key = _normalize(sampler, SUPPORTED_SAMPLERS)
    scheduler_key = _normalize(scheduler, SUPPORTED_SCHEDULERS)
    shift_value = _coerce_shift(shift) if scheduler_key == "shift" else None

    if sampler_key == "default" and scheduler_key == "default":
        # Pure no-op path — but if a prior call swapped the scheduler on this
        # cached pipe, revert so the user's "default" choice means default.
        restored = _restore_original(pipe)
        return {
            "sampler": "default",
            "scheduler": "default",
            "noop": True,
            "restored": restored,
        }

    # Route to the spec table matching the pipe's prediction family so SDXL
    # (epsilon) pipes get standard solvers and flow pipes keep their flow-mode
    # solvers. A sampler key valid in one family but absent from the other
    # (e.g. ``euler_a`` / ``dpmpp_sde`` are standard-only) degrades to "keep the
    # current class, apply the scheduler axis only" rather than hard-failing.
    family = _scheduler_family(pipe)
    samplers_table = _STD_SAMPLERS if family == "standard" else _FLOW_SAMPLERS
    if sampler_key in samplers_table:
        spec = samplers_table[sampler_key]
    else:
        spec = None
        if sampler_key != "default":
            _emit(
                "sampler_unavailable_for_family",
                adapter=adapter,
                sampler=sampler_key,
                family=family,
            )
    sigma_flags = _SCHEDULER_SIGMA_FLAGS.get(scheduler_key, {})

    # Build the target class. When sampler == "default" we keep whatever the
    # pipe currently has (or restore the original) and just re-apply the
    # scheduler flags on top.
    if spec is None:
        _restore_original(pipe)
        target_cls = type(pipe.scheduler)
    else:
        try:
            diffusers = importlib.import_module("diffusers")
        except ImportError:
            _emit(
                "sampler_apply_skipped",
                adapter=adapter,
                reason="diffusers_import_failed",
                sampler=sampler_key,
                scheduler=scheduler_key,
            )
            return {"sampler": "default", "scheduler": "default", "noop": True}
        target_cls = getattr(diffusers, spec.scheduler_cls, None)
        if target_cls is None:
            _emit(
                "sampler_apply_skipped",
                adapter=adapter,
                reason="scheduler_class_unavailable",
                sampler=sampler_key,
                scheduler=scheduler_key,
                expectedClass=spec.scheduler_cls,
            )
            return {"sampler": "default", "scheduler": "default", "noop": True}

    overrides: dict[str, Any] = {}
    if spec is not None:
        overrides.update(spec.config_overrides)
    overrides.update(sigma_flags)
    if shift_value is not None:
        overrides["shift"] = shift_value
        # Diffusers' FlowMatch variants use `use_dynamic_shifting`. When the
        # user explicitly pins shift, turn dynamic shifting off so the value
        # actually applies.
        overrides.setdefault("use_dynamic_shifting", False)

    filtered, dropped = _filter_overrides(target_cls, overrides)
    base_config = getattr(pipe.scheduler, "config", None)

    try:
        if base_config is not None:
            new_scheduler = target_cls.from_config(base_config, **filtered)
        else:
            new_scheduler = target_cls(**filtered)
    except Exception as exc:  # noqa: BLE001
        _emit(
            "sampler_apply_failed",
            adapter=adapter,
            sampler=sampler_key,
            scheduler=scheduler_key,
            shift=shift_value,
            schedulerClass=getattr(target_cls, "__name__", None),
            droppedFlags=dropped,
            error=str(exc),
            errorType=exc.__class__.__name__,
        )
        return {
            "sampler": "default",
            "scheduler": "default",
            "noop": True,
            "fallback": True,
        }

    pipe.scheduler = new_scheduler
    pipe._sceneworks_sampler_dirty = True
    mu_shim_installed = False
    if not _set_timesteps_accepts_mu(new_scheduler):
        mu_shim_installed = _install_mu_shim(new_scheduler, adapter=adapter)
    _emit(
        "sampler_applied",
        adapter=adapter,
        sampler=sampler_key,
        scheduler=scheduler_key,
        family=family,
        shift=shift_value,
        schedulerClass=getattr(target_cls, "__name__", None),
        droppedFlags=dropped,
        appliedFlags=sorted(filtered),
        muShimInstalled=mu_shim_installed,
    )
    return {
        "sampler": sampler_key,
        "scheduler": scheduler_key,
        "family": family,
        "shift": shift_value,
        "schedulerClass": getattr(target_cls, "__name__", None),
        "appliedFlags": sorted(filtered),
        "droppedFlags": dropped,
        "muShimInstalled": mu_shim_installed,
        "noop": False,
    }


def sampler_selection_from_advanced(advanced: dict[str, Any]) -> tuple[str, str, float | None]:
    """Extract the (sampler, scheduler, shift) triple from a job's advanced dict.

    Returns normalized defaults when fields are missing or invalid. Shift is
    ``None`` unless ``scheduler == "shift"`` and ``schedulerShift`` parses as
    a positive float.
    """
    if not isinstance(advanced, dict):
        return ("default", "default", None)
    sampler_key = _normalize(advanced.get("sampler"), SUPPORTED_SAMPLERS)
    scheduler_key = _normalize(advanced.get("scheduler"), SUPPORTED_SCHEDULERS)
    if scheduler_key == "shift":
        shift_value = _coerce_shift(advanced.get("schedulerShift"))
    else:
        shift_value = None
    return (sampler_key, scheduler_key, shift_value)
