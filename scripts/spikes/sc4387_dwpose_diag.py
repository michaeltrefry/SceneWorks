#!/usr/bin/env python3
"""sc-4387 — DWPose backend spike diagnostics (does CoreML belong? + op inventory + latency).

Runs three things against the two production ONNX exports (yolox_m detector +
rtmw-dw-x-l SimCC pose net) on this Apple Silicon machine:

  1. Op-type histogram + param/initializer counts per graph  -> grounds the
     native-MLX feasibility/effort call (which ops mlx_gen::nn covers vs bespoke
     mlx_rs::ops; how many MatMul/Gemm are reduced-precision-risk on Metal).
  2. CoreML EP partition diagnostic (SAM2-style): how many nodes CoreML accepts
     and into how many partitions it splits the graph -> answers "does CoreML
     even belong here?" (fragmentation = net negative, cf. sc-3635 SAM2).
  3. CoreML vs CPU steady-state latency on correctly-shaped random inputs ->
     confirms / refreshes sc-3487's CoreML 3-5x-over-CPU finding on this box.

Usage:
  ~/.dwpose-spike/venv/bin/python scripts/spikes/sc4387_dwpose_diag.py
"""

from __future__ import annotations

import collections
import io
import os
import re
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

CKPT = Path.home() / ".cache/rtmlib/hub/checkpoints"
DET = CKPT / "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
POSE = CKPT / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"


def human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


def inventory(path: Path) -> None:
    m = onnx.load(str(path))
    g = m.graph
    ops = collections.Counter(n.op_type for n in g.node)
    # initializer param count
    params = 0
    for init in g.initializer:
        size = 1
        for d in init.dims:
            size *= d
        params += size
    print(f"\n=== {path.name} ===")
    print(f"  nodes: {len(g.node)}   params: {human(params)}")
    print("  inputs:")
    for i in g.input:
        shp = [d.dim_value if (d.dim_value or 0) > 0 else d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"    {i.name}: {shp}")
    print("  outputs:")
    for o in g.output:
        shp = [d.dim_value if (d.dim_value or 0) > 0 else d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"    {o.name}: {shp}")
    print("  op histogram:")
    for op, c in ops.most_common():
        print(f"    {op:24s} {c}")


def partitions(path: Path) -> None:
    """Create a CoreML-EP session with VERBOSE logging and surface the partition
    summary onnxruntime prints from GetCapability.

    onnxruntime's C++ log goes to the process stderr (fd 2). We redirect fd 2 to a
    temp FILE (not a pipe — a pipe's 64KB buffer fills during verbose partition
    logging and deadlocks session creation), then read it back."""
    import tempfile

    so = ort.SessionOptions()
    so.log_severity_level = 0  # VERBOSE -> partition summary goes to stderr
    print(f"\n=== CoreML partitioning: {path.name} ===")
    captured = ""
    with tempfile.NamedTemporaryFile("w+", suffix=".log") as tf:
        old = os.dup(2)
        os.dup2(tf.fileno(), 2)
        try:
            sess = ort.InferenceSession(
                str(path),
                sess_options=so,
                providers=[("CoreMLExecutionProvider", {})],
            )
            sys.stderr.flush()
            os.fsync(tf.fileno())
        except Exception as e:  # noqa: BLE001
            os.dup2(old, 2)
            os.close(old)
            print(f"  CoreML session failed: {e}")
            return
        finally:
            os.dup2(old, 2)
            try:
                os.close(old)
            except OSError:
                pass
        tf.seek(0)
        captured = tf.read()
    print(f"  providers in use: {sess.get_providers()}")
    # Pull the lines that report node/partition placement.
    hits = 0
    for line in captured.splitlines():
        if re.search(r"partition|placed on|number of nodes|CoreML", line, re.I):
            print("   ", line.split("]", 1)[-1].strip()[:200])
            hits += 1
    if hits == 0:
        print(f"  (no placement lines matched; captured {len(captured)} bytes of log)")


def latency(path: Path, shape: tuple, tag: str, runs: int = 20) -> None:
    print(f"\n=== latency: {path.name} ===")
    for ep_name, ep in (("CPU", "CPUExecutionProvider"), ("CoreML", "CoreMLExecutionProvider")):
        try:
            sess = ort.InferenceSession(str(path), providers=[ep])
        except Exception as e:  # noqa: BLE001
            print(f"  {ep_name}: session failed: {e}")
            continue
        # use the graph's real input name (don't assume "input")
        feeds = {inp.name: np.random.rand(*shape).astype(np.float32) for inp in sess.get_inputs()}
        try:
            # warmup (CoreML compiles the graph on first run)
            t0 = time.perf_counter()
            sess.run(None, feeds)
            warm = (time.perf_counter() - t0) * 1e3
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                sess.run(None, feeds)
                times.append((time.perf_counter() - t0) * 1e3)
        except Exception as e:  # noqa: BLE001
            # yolox's embedded NMS yields 0 boxes on random noise -> CoreML rejects the
            # zero-element dynamic tensor. Real images have people; benign for latency.
            print(f"  {ep_name:7s} run failed on random input ({str(e).splitlines()[0][:80]})")
            continue
        times.sort()
        med = times[len(times) // 2]
        print(f"  {ep_name:7s} warmup {warm:7.1f}ms  median {med:7.1f}ms  min {times[0]:7.1f}ms  (n={runs})")


def main() -> None:
    print(f"onnxruntime {ort.__version__}   providers {ort.get_available_providers()}")
    for p in (DET, POSE):
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            sys.exit(1)
    inventory(DET)
    inventory(POSE)
    partitions(DET)
    partitions(POSE)
    # yolox_m input is dynamic-batch (1,3,640,640); rtmw is (1,3,384,288)
    latency(DET, (1, 3, 640, 640), "det")
    latency(POSE, (1, 3, 384, 288), "pose")


if __name__ == "__main__":
    main()
