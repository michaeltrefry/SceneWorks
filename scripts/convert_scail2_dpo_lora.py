"""sc-5451 / sc-5686: convert SCAIL-2's Bias-Aware DPO LoRA to an engine-loadable safetensors.

zai-org/SCAIL-2 ships its Bias-Aware DPO refinement LoRA as `model/bias-aware-dpo-lora.pt`
(in `SceneWorks/scail2-mlx` and the upstream HF snapshot). That file is a **DeepSpeed/SAT**
(SwissArmyTransformer) checkpoint — the LoRA state lives under the top-level `["module"]` key with
SAT module names and `.lora_layer[.{idx}].{down,up}.weight` factors — which the native MLX engine's
LoRA loader (which reads a `.safetensors` of standard `lora_down`/`lora_up` factors named for the
inference `SCAIL2Model` modules) cannot consume directly.

This converts it to `scail2-dpo-lora.safetensors`: a plain rank-128 LoRA whose keys are the inference
module names the engine resolves (`blocks.N.{self_attn,cross_attn}.{q,k,v,o}`, `blocks.N.ffn.{0,2}`),
with the SAT fused projections split out per sub-LoRA index (`query_key_value.lora_layer.{0,1,2}` =
q/k/v; `key_value.lora_layer.{0,1}` = k/v; `dense` = the output proj `o`; `mlp.dense_h_to_4h`/`4h_to_h`
= `ffn.0`/`ffn.2`). The mapping is a pure rename — the per-projection sub-LoRAs are already separate,
and the dims confirm it (e.g. `mlp.dense_h_to_4h` down[128,5120]/up[13824,128] == SCAIL-2's
`ffn.0` [13824,5120]). NO diff-patch tensors (unlike the lightx2v lightning LoRAs, sc-5684).

The output applies through the family-agnostic engine loader (`apply_adapters_strict`) as a forward-time
residual over the Q4 base — validated 2026-06-14 on a 128 GB Mac (400 modules matched, 16-frame
generation, CFG-on). Host the result alongside the model in `SceneWorks/scail2-mlx` and catalog it in
`config/manifests/builtin.loras.jsonc`.

Usage:
    python scripts/convert_scail2_dpo_lora.py <bias-aware-dpo-lora.pt> <scail2-dpo-lora.safetensors>
"""

import re
import sys

import torch
from safetensors.torch import save_file

# SAT (SwissArmyTransformer) module → inference `SCAIL2Model` module. The fused `query_key_value` /
# `key_value` projections carry one `lora_layer.{idx}` per split (q/k/v, k/v); the rest are bare.
SAT_TO_INFERENCE = {
    "attention.query_key_value.lora_layer.0": "self_attn.q",
    "attention.query_key_value.lora_layer.1": "self_attn.k",
    "attention.query_key_value.lora_layer.2": "self_attn.v",
    "attention.dense.lora_layer": "self_attn.o",
    "cross_attention.query.lora_layer": "cross_attn.q",
    "cross_attention.key_value.lora_layer.0": "cross_attn.k",
    "cross_attention.key_value.lora_layer.1": "cross_attn.v",
    "cross_attention.dense.lora_layer": "cross_attn.o",
    "mlp.dense_h_to_4h.lora_layer": "ffn.0",
    "mlp.dense_4h_to_h.lora_layer": "ffn.2",
}

_KEY = re.compile(
    r"^model\.diffusion_model\.transformer\.layers\.(\d+)\.(.+)\.(down|up)\.weight$"
)


def convert(src_pt: str, out_safetensors: str) -> None:
    ckpt = torch.load(src_pt, map_location="cpu", weights_only=False)
    # DeepSpeed wraps the trained params under "module"; tolerate a bare state dict too.
    sd = ckpt["module"] if isinstance(ckpt, dict) and "module" in ckpt else ckpt

    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if not hasattr(value, "shape"):
            continue  # DeepSpeed RNG/optimizer bookkeeping
        m = _KEY.match(key)
        if not m:
            raise SystemExit(f"unexpected SAT LoRA key (not a layer down/up factor): {key}")
        layer, sat_module, factor = m.group(1), m.group(2), m.group(3)
        inference_module = SAT_TO_INFERENCE.get(sat_module)
        if inference_module is None:
            raise SystemExit(f"unmapped SAT module '{sat_module}' (key {key})")
        new_factor = "lora_down" if factor == "down" else "lora_up"
        out_key = f"blocks.{layer}.{inference_module}.{new_factor}.weight"
        out[out_key] = value.to(torch.bfloat16).contiguous()

    save_file(out, out_safetensors)
    pairs = len(out) // 2
    print(f"converted {len(out)} tensors → {pairs} LoRA pairs → {out_safetensors}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    convert(sys.argv[1], sys.argv[2])
