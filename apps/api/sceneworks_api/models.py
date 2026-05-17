from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request


router = APIRouter(prefix="/models", tags=["models"])


def strip_jsonc_comments(value: str) -> str:
    value = re.sub(r"//.*", "", value)
    return re.sub(r"/\*.*?\*/", "", value, flags=re.S)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.loads(strip_jsonc_comments(handle.read()))
    return payload.get("models", [])


@router.get("")
def list_models(request: Request) -> list[dict[str, Any]]:
    settings = request.app.state.settings
    manifest_dir = settings.config_dir / "manifests"
    builtin = load_manifest(manifest_dir / "builtin.models.jsonc")
    user = load_manifest(manifest_dir / "user.models.jsonc")
    by_id = {model["id"]: model for model in builtin if "id" in model}
    for model in user:
        model_id = model.get("id")
        if model_id:
            by_id[model_id] = {**by_id.get(model_id, {}), **model}
    return sorted(by_id.values(), key=lambda model: (model.get("type", ""), model.get("name", "")))
