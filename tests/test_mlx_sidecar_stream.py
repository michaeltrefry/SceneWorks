"""Streaming behaviour of the shared MLX sidecar controller (sc-2412).

``_MlxSidecarStream`` is what makes the mflux families (FLUX.1 / FLUX.2 / Qwen /
Z-Image) surface each image as it's produced instead of all at once when the
sidecar exits. These tests drive the real controller against a tiny stand-in
``runner`` script (plain ``sys.executable`` — no mflux / MLX needed) so the
streaming, fallback, failure, and cancel paths are exercised end-to-end through
a real subprocess + the reader thread, not mocks.
"""
from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from scene_worker import image_adapters as ia


def _spec(work: Path, n: int) -> Path:
    spec_path = work / "spec.json"
    spec_path.write_text(
        json.dumps({"outDir": str(work), "seeds": list(range(1, n + 1)), "model": "flux_schnell"}),
        encoding="utf-8",
    )
    return spec_path


def _stream(work: Path, runner: Path, total: int, *, cancel=lambda: False) -> ia._MlxSidecarStream:
    return ia._MlxSidecarStream(
        cmd=[sys.executable, str(runner), str(work / "spec.json")],
        job_id="job_stream",
        adapter_id="mlx_flux",
        work_dir=work,
        stdout_log=work / "stdout.log",
        total=total,
        cancel_requested=cancel,
        read_result=ia.MlxFluxAdapter._read_result,
        complete_event="mlx_flux_sidecar_complete",
        failed_event="mlx_flux_sidecar_failed",
        fail_label="MLX FLUX",
    ).start()


def test_stream_surfaces_each_image_before_the_process_exits(tmp_path):
    """The crux of sc-2412: image 0 must be returned by wait_for_image WHILE the
    sidecar is still running (gated on later images), proving per-image streaming
    rather than the old all-at-once-at-exit behaviour."""
    runner = tmp_path / "runner_stream.py"
    runner.write_text(
        textwrap.dedent(
            """
            import json, sys, time
            from pathlib import Path

            spec = json.loads(Path(sys.argv[1]).read_text())
            out = Path(spec["outDir"])
            images = []
            for i in range(len(spec["seeds"])):
                p = out / f"img_{i:04d}.png"
                p.write_bytes(b"placeholder")
                images.append(str(p))
                print(json.dumps({"event": "image", "index": i, "path": str(p)}), flush=True)
                # Block until the test releases this image, so it can observe that
                # earlier images are already consumable while the run continues.
                gate = out / f"gate_{i}"
                while not gate.exists():
                    time.sleep(0.01)
            (out / "result.json").write_text(json.dumps({"images": images}))
            print(json.dumps({"images": images}), flush=True)
            """
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    _spec(work, 3)

    # Safety net: if anything hangs, cancel after 30s so the test fails instead of
    # blocking CI forever.
    start = time.monotonic()
    stream = _stream(work, runner, 3, cancel=lambda: time.monotonic() - start > 30)
    try:
        first = stream.wait_for_image(0)
        assert first.endswith("img_0000.png")
        # Still running (gated on gate_0) — image 0 surfaced mid-flight.
        assert stream._proc.poll() is None

        (work / "gate_0").touch()
        assert stream.wait_for_image(1).endswith("img_0001.png")
        (work / "gate_1").touch()
        assert stream.wait_for_image(2).endswith("img_0002.png")
        (work / "gate_2").touch()

        stream.finish()  # validates count + clean exit
    finally:
        stream.shutdown()


def test_stream_falls_back_to_result_json_without_markers(tmp_path):
    """Graceful degradation: a runner that emits no per-image markers still works
    — wait_for_image falls back to the authoritative result.json ordering once the
    process exits (i.e. the old all-at-once behaviour). Correctness never depends
    on the markers."""
    runner = tmp_path / "runner_silent.py"
    runner.write_text(
        textwrap.dedent(
            """
            import json, sys
            from pathlib import Path

            spec = json.loads(Path(sys.argv[1]).read_text())
            out = Path(spec["outDir"])
            images = []
            for i in range(len(spec["seeds"])):
                p = out / f"img_{i:04d}.png"
                p.write_bytes(b"placeholder")
                images.append(str(p))
            (out / "result.json").write_text(json.dumps({"images": images}))
            print(json.dumps({"images": images}), flush=True)
            """
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    _spec(work, 3)

    stream = _stream(work, runner, 3)
    try:
        assert stream.wait_for_image(0).endswith("img_0000.png")
        assert stream.wait_for_image(2).endswith("img_0002.png")
        stream.finish()
    finally:
        stream.shutdown()


def test_stream_surfaces_sidecar_failure(tmp_path):
    """A non-zero exit with an error result must raise the same enriched error the
    old blocking _run_sidecar raised."""
    runner = tmp_path / "runner_fail.py"
    runner.write_text(
        textwrap.dedent(
            """
            import json, sys
            from pathlib import Path

            spec = json.loads(Path(sys.argv[1]).read_text())
            out = Path(spec["outDir"])
            (out / "result.json").write_text(json.dumps({"error": "boom"}))
            print(json.dumps({"error": "boom"}), flush=True)
            sys.exit(1)
            """
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    _spec(work, 2)

    stream = _stream(work, runner, 2)
    try:
        with pytest.raises(RuntimeError, match="MLX FLUX generation failed"):
            stream.wait_for_image(0)
    finally:
        stream.shutdown()


def test_runner_emit_image_marker_shape(capsys):
    """Lock the producer side of the contract: mlx_flux_runner._emit_image must
    print exactly the marker shape _MlxSidecarStream parses. (Importing the runner
    is safe — it only pulls json/sys/pathlib at module scope; mflux is lazy.)"""
    from scene_worker import mlx_flux_runner as runner

    runner._emit_image(2, Path("/work/img_0002.png"))
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {"event": "image", "index": 2, "path": "/work/img_0002.png"}


def test_stream_cancel_raises_and_terminates(tmp_path):
    """Cancellation while waiting raises InterruptedError and kills the sidecar."""
    runner = tmp_path / "runner_sleep.py"
    runner.write_text(
        textwrap.dedent(
            """
            import time
            # Never produces an image; just hangs until terminated.
            while True:
                time.sleep(0.05)
            """
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    _spec(work, 1)

    cancelled = {"flag": False}
    stream = _stream(work, runner, 1, cancel=lambda: cancelled["flag"])
    timer = threading.Timer(0.3, lambda: cancelled.__setitem__("flag", True))
    timer.start()
    try:
        with pytest.raises(InterruptedError):
            stream.wait_for_image(0)
        assert stream._proc.poll() is not None  # terminated
    finally:
        timer.cancel()
        stream.shutdown()
