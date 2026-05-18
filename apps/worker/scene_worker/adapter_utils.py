from __future__ import annotations

import inspect
from typing import Any


def accepted_call_parameters(callable_owner: Any) -> set[str]:
    try:
        return set(inspect.signature(callable_owner.__call__).parameters)
    except (TypeError, ValueError):
        return set()


def filter_call_kwargs(callable_owner: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    accepted = accepted_call_parameters(callable_owner)
    if not accepted:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}
