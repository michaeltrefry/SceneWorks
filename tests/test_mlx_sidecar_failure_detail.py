"""Unit tests for ``_mlx_sidecar_failure_detail`` (sc-2274).

The MLX sidecars run out-of-process; when the OS kills one with a signal
(SIGKILL on memory pressure, SIGABRT/SIGSEGV on a native Metal fault) it
bypasses ``mlx_flux_runner``'s ``try/except`` and the only structured error
is the opaque "produced no parseable result". ``_mlx_sidecar_failure_detail``
turns the (negative) return code + any captured output into an actionable
message. These tests pin that behavior.
"""

from __future__ import annotations

from scene_worker.image_adapters import _mlx_sidecar_failure_detail

BASE = "MLX FLUX sidecar produced no parseable result."


def test_sigkill_is_reported_as_out_of_memory():
    msg = _mlx_sidecar_failure_detail(BASE, -9, _missing_log())
    assert BASE in msg
    assert "SIGKILL" in msg
    assert "out of memory" in msg
    # Actionable levers are surfaced to the user.
    assert "Q4" in msg
    assert "resolution" in msg
    assert "LoRA" in msg


def test_sigabrt_is_reported_as_native_fault():
    msg = _mlx_sidecar_failure_detail(BASE, -6, _missing_log())
    assert "SIGABRT" in msg
    assert "native MLX/Metal fault" in msg
    assert "out of memory" not in msg


def test_sigsegv_is_reported_as_native_fault():
    msg = _mlx_sidecar_failure_detail(BASE, -11, _missing_log())
    assert "SIGSEGV" in msg
    assert "native MLX/Metal fault" in msg


def test_unknown_signal_uses_generic_label():
    msg = _mlx_sidecar_failure_detail(BASE, -15, _missing_log())
    assert "signal 15" in msg


def test_clean_nonzero_exit_is_unchanged():
    # A positive return code means the runner exited normally and already wrote
    # a structured error; we must not invent a signal explanation.
    msg = _mlx_sidecar_failure_detail("some structured error", 1, _missing_log())
    assert msg == "some structured error"


def test_captured_output_tail_is_appended(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("noise\nRuntimeError: kaboom\n", encoding="utf-8")
    msg = _mlx_sidecar_failure_detail(BASE, -6, log)
    assert "last sidecar output:" in msg
    assert "RuntimeError: kaboom" in msg


def test_output_tail_is_limited_to_last_15_lines(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("\n".join(f"line{i}" for i in range(40)) + "\n", encoding="utf-8")
    msg = _mlx_sidecar_failure_detail(BASE, 1, log)
    assert "line39" in msg  # newest line kept
    assert "line25" in msg  # 40 lines -> tail is line25..line39
    assert "line24" not in msg  # everything older is dropped
    assert "line0" not in msg


def test_no_error_and_no_output_has_a_fallback(tmp_path):
    msg = _mlx_sidecar_failure_detail("", 0, tmp_path / "missing.log")
    assert msg == "MLX sidecar failed with no output."


def _missing_log():
    from pathlib import Path

    return Path("/nonexistent/stdout.log")
