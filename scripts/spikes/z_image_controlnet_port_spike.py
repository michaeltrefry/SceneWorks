#!/usr/bin/env python3
"""sc-2257 port spike — step 1: confirm the Z-Image Fun-Controlnet-Union weight
layout against the expected MLX mapping before any module gets written.

WHY THIS EXISTS
---------------
Porting `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` into the mflux (Apple
MLX) Z-Image stack is a VACE-style ControlNet port (NOT the simple FLUX zero-init
Linear template): 15 control transformer blocks + 2 control refiners, a 33-channel
VAE-encoded control context, before_proj/after_proj stack-threading, and dual
injection sites (noise_refiner + main loop). Highest-risk correctness item is
padding / RoPE alignment, which can only be validated numerically on real hardware.

This script is the cheap FIRST step of the recommended spike sequence: open the
ControlNet safetensors and confirm its key layout matches what an MLX
`WeightMapping` will need, so the real port targets the right module paths. It does
NOT load mflux or MLX — just `safetensors` + the weight file.

RUN (on the Mac, in any venv with `safetensors` + `huggingface_hub`):
    python scripts/spikes/z_image_controlnet_port_spike.py \
        --weights /path/to/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors

If you don't have the file yet:
    huggingface-cli download alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
        Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors --local-dir ./zic

Reference (from the sc-2257 spike research):
    - mflux Z-Image transformer: models/z_image/model/z_image_transformer/transformer.py
      (injection loop transformer.py:122-128; existing base mapping
      z_image_weight_mapping.py)
    - torch CN reference: VideoX-Fun videox_fun/models/z_image_transformer2d_control.py
    - config: VideoX-Fun config/z_image/z_image_control_2.1.yaml
      control_layers_places=[0,2,4,...,28] (15), control_refiner_layers_places=[0,1],
      control_in_dim=33, add_control_noise_refiner=true
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict


# Expected NEW control-branch key groups an MLX loader must map (base Z-Image keys
# such as layers.*/noise_refiner.*/context_refiner.* are already mapped in mflux's
# z_image_weight_mapping.py and are NOT re-listed here).
EXPECTED_CONTROL_GROUPS = {
    "control_all_x_embedder": "33ch->dim control patch embedder (zero-init Linear, key 2-1)",
    "control_layers": "15 parallel control transformer blocks (places 0,2,...,28)",
    "control_noise_refiner": "2 control refiner blocks (add_control_noise_refiner=True)",
}
# Per-control-block submodules expected inside control_layers.{n}.* (mirrors a base
# ZImageTransformerBlock) plus the VACE projections.
EXPECTED_BLOCK_SUBKEYS = [
    "attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0",
    "attention.norm_q", "attention.norm_k",
    "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2",
    "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
    "adaLN_modulation.0",
    "before_proj",  # only on control_layers.0 / control_noise_refiner.0
    "after_proj",   # on every control block
]


def _top_group(key: str) -> str:
    return key.split(".", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="path to the Fun-Controlnet-Union safetensors")
    args = parser.parse_args()

    try:
        from safetensors import safe_open
    except ImportError:
        raise SystemExit("pip install safetensors (and huggingface_hub to fetch the weights)")

    keys: list[str] = []
    shapes: dict[str, tuple] = {}
    with safe_open(args.weights, framework="numpy") as f:  # framework only affects get_tensor; keys are framework-agnostic
        for k in f.keys():
            keys.append(k)
            try:
                shapes[k] = tuple(f.get_slice(k).get_shape())
            except Exception:
                shapes[k] = ()

    print(f"\n=== {len(keys)} tensors in {args.weights} ===\n")

    # 1. Top-level group census.
    groups = Counter(_top_group(k) for k in keys)
    print("Top-level key groups:")
    for g, n in sorted(groups.items()):
        flag = "  <-- NEW control group" if g in EXPECTED_CONTROL_GROUPS else ""
        print(f"  {g:<28} {n:>5} tensors{flag}")

    # 2. Confirm each expected NEW control group is present.
    print("\nExpected control groups (vs §4 of the sc-2257 map):")
    for g, desc in EXPECTED_CONTROL_GROUPS.items():
        present = "OK " if g in groups else "MISSING"
        print(f"  [{present}] {g:<24} {desc}")

    # 3. control_layers.N index coverage — expect 15 contiguous (0..14).
    layer_idx = sorted({int(m.group(1)) for k in keys if (m := re.match(r"control_layers\.(\d+)\.", k))})
    print(f"\ncontrol_layers indices present: {layer_idx}")
    print(f"  count={len(layer_idx)} (expect 15 for v2.1)  "
          f"contiguous={'yes' if layer_idx == list(range(len(layer_idx))) else 'NO'}")
    refiner_idx = sorted({int(m.group(1)) for k in keys if (m := re.match(r"control_noise_refiner\.(\d+)\.", k))})
    print(f"control_noise_refiner indices: {refiner_idx} (expect [0, 1])")

    # 4. before_proj / after_proj placement (VACE stack-threading projections).
    before = sorted(k.rsplit(".", 1)[0] for k in keys if ".before_proj." in k or k.endswith(".before_proj.weight"))
    after_blocks = sorted({m.group(0) for k in keys if (m := re.match(r"control_layers\.\d+", k)) and ".after_proj." in k})
    print(f"\nbefore_proj on: {before}  (expect ONLY control_layers.0 + control_noise_refiner.0)")
    print(f"after_proj present on control_layers: {len(after_blocks)} blocks (expect 15)")

    # 5. control_all_x_embedder shape — confirms the 33ch input dim.
    emb_keys = [k for k in keys if k.startswith("control_all_x_embedder")]
    print("\ncontrol_all_x_embedder tensors (confirms control_in_dim=33 via in-features):")
    for k in emb_keys:
        print(f"  {k}  shape={shapes.get(k)}")

    # 6. Per-block submodule coverage on control_layers.0 (the richest block).
    block0 = [k for k in keys if k.startswith("control_layers.0.")]
    print("\ncontrol_layers.0 submodules found:")
    found_sub = defaultdict(list)
    for k in block0:
        inner = k[len("control_layers.0."):]
        found_sub[inner.rsplit(".", 1)[0]].append(k)
    for sub in EXPECTED_BLOCK_SUBKEYS:
        hit = any(s.startswith(sub) for s in found_sub)
        print(f"  [{'OK ' if hit else '???'}] {sub}")

    print("\n=== Next step (sequence item 2): clone ZImageTransformerBlock -> "
          "ZImageControlTransformerBlock (MLX) + zero-init after_proj/before_proj, "
          "then add control_layers + control_all_x_embedder + the 33ch control_context "
          "VAE-encode path. See the sc-2257 story comment for the full map. ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
