"""Single source of truth for resolving Hugging Face cache paths.

This consolidates what were three near-identical copies in the image, lora, and
video adapters. The lora copy skipped the ``resolve()`` + ``relative_to()``
traversal guard and consulted a different cache-root order, so a repo cached
under one root was "found here but not there". Everything now flows through one
guarded primitive (:func:`huggingface_repo_cache_path_for_root`) and one ordered,
de-duplicated list of roots (:func:`huggingface_cache_roots`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def safe_repo_dir_name(repo: str) -> str | None:
    """The ``models--<repo>`` directory stem HF uses, unsafe chars folded to ``--``."""

    name = "".join(char if char.isalnum() or char in "._-" else "--" for char in repo).strip("-")
    return name or None


def huggingface_cache_root() -> Path:
    """The primary HF hub cache root from the environment (falls back to ~/.cache)."""

    default_home = Path.home() / ".cache" / "huggingface"
    hf_home = Path(os.getenv("HF_HOME") or default_home)
    return Path(os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE") or hf_home / "hub")


def huggingface_cache_roots(settings: Any | None = None) -> list[Path]:
    """All HF hub cache roots to consult, in priority order, de-duplicated.

    Order: explicit ``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE``, then ``HF_HOME/hub``,
    then the app data dir's cache (``settings.data_dir`` if given, else
    ``$SCENEWORKS_DATA_DIR``), then ``huggingface_cache_root()`` (env-or-~/.cache).
    """

    roots: list[Path] = []
    for value in (os.getenv("HF_HUB_CACHE"), os.getenv("HUGGINGFACE_HUB_CACHE")):
        if value:
            roots.append(Path(value).expanduser())
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    data_dir = getattr(settings, "data_dir", None)
    if data_dir is None:
        data_dir = Path(os.getenv("SCENEWORKS_DATA_DIR", "data")).expanduser()
    roots.append(Path(data_dir) / "cache" / "huggingface" / "hub")
    roots.append(huggingface_cache_root())
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def huggingface_repo_cache_path_for_root(cache_root: Path, repo: str) -> Path | None:
    """``<cache_root>/models--<repo>``, guarded so it can never escape ``cache_root``."""

    safe_repo = safe_repo_dir_name(repo)
    if not safe_repo:
        return None
    try:
        root = cache_root.resolve()
        repo_cache = (root / f"models--{safe_repo}").resolve()
        repo_cache.relative_to(root)
    except (OSError, ValueError):
        return None
    return repo_cache


def huggingface_repo_cache_path(repo: str, settings: Any | None = None) -> Path | None:
    """The cached ``models--<repo>`` directory.

    Returns the first known root where it actually exists; if it exists nowhere,
    returns the guarded path under the primary root (for callers asking "where
    would it go"). Returns ``None`` only when ``repo`` has no safe directory name.
    """

    fallback: Path | None = None
    for root in huggingface_cache_roots(settings):
        repo_cache = huggingface_repo_cache_path_for_root(root, repo)
        if repo_cache is None:
            continue
        if fallback is None:
            fallback = repo_cache
        if repo_cache.exists():
            return repo_cache
    return fallback
