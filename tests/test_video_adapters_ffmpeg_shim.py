from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from scene_worker import video_adapters


def _make_fake_ffmpeg(tmp_path: Path) -> Path:
    exe = tmp_path / "bundled" / "ffmpeg"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    return exe


def test_ensure_ffmpeg_on_path_uses_private_unpredictable_dir(tmp_path, monkeypatch):
    """The ffmpeg shim dir must be a private, per-process mkdtemp dir (mode 0700,
    unpredictable name) — not a fixed, world-traversable name in the shared temp
    root that another local user could pre-plant a malicious symlink in
    (sc-4189 / F-WORKER-5)."""
    exe = _make_fake_ffmpeg(tmp_path)

    # No system ffmpeg; bundled exe supplied via env.
    monkeypatch.setenv("SCENEWORKS_FFMPEG", str(exe))
    # Point the temp root at our sandbox so we can assert on what gets created.
    fake_tmp = tmp_path / "tmproot"
    fake_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(fake_tmp))
    monkeypatch.setattr("tempfile.tempdir", None, raising=False)

    # Strip any existing ffmpeg from PATH so the function does real work.
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))

    video_adapters._ensure_ffmpeg_on_path()

    path_entries = os.environ["PATH"].split(os.pathsep)
    shim_dir = Path(path_entries[0])

    # The shim dir is the freshly-prepended entry, lives under our temp root,
    # and has an unpredictable (non-fixed) name.
    assert shim_dir.parent == fake_tmp
    assert shim_dir.name.startswith("sceneworks-ffmpeg-shim-")
    assert shim_dir.name != "sceneworks-ffmpeg-shim"

    # Private to the owner: mkdtemp yields 0700.
    mode = stat.S_IMODE(shim_dir.stat().st_mode)
    assert mode == 0o700, f"expected 0700, got {oct(mode)}"

    # The symlink resolves to the bundled exe.
    link = shim_dir / "ffmpeg"
    assert link.is_symlink()
    assert link.resolve() == exe.resolve()


def test_ensure_ffmpeg_on_path_noop_when_ffmpeg_present(tmp_path, monkeypatch):
    """If a real ffmpeg is already discoverable on PATH, no shim dir is created
    and PATH is left untouched."""
    real_dir = tmp_path / "realbin"
    real_dir.mkdir()
    real_ffmpeg = real_dir / "ffmpeg"
    real_ffmpeg.write_text("#!/bin/sh\nexit 0\n")
    real_ffmpeg.chmod(0o755)

    monkeypatch.delenv("SCENEWORKS_FFMPEG", raising=False)
    monkeypatch.setenv("PATH", str(real_dir))
    before = os.environ["PATH"]

    video_adapters._ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == before
