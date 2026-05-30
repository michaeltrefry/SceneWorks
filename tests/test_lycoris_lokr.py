"""Third-party LyCORIS LoKr/LoHa adapter handling (epic 2193).

Two layers:
  * Pure-logic ``classify_adapter_network`` tests (no torch) — run in lightweight CI.
  * A torch + ``lycoris`` round-trip through the production
    ``apply_loras_to_pipeline`` / ``clear_loras`` path (skipped without the deps).

Background: kohya / ai-toolkit LyCORIS adapters carry ``lokr_w1``/``lokr_w2`` (or
``hada_*``) tensors that diffusers' ``load_lora_weights`` cannot consume — without
detection they crash with a multi-thousand-line "state_dict should be empty at this
point" dump. These tests pin that they're detected and applied via the lycoris
loader instead.
"""

from __future__ import annotations

import pytest

from scene_worker import lora_adapters
from scene_worker.lora_adapters import LoraSpec, classify_adapter_network


# --------------------------------------------------------------------------- #
# Pure-logic classification (no torch / no lycoris)
# --------------------------------------------------------------------------- #


def _patch_adapter(monkeypatch, *, metadata=None, tensor_names=()):
    monkeypatch.setattr(lora_adapters, "read_adapter_metadata", lambda _p: dict(metadata or {}))
    monkeypatch.setattr(lora_adapters, "_adapter_tensor_names", lambda _p, limit=256: list(tensor_names))


def test_sceneworks_peft_lokr_classified_as_lokr(monkeypatch):
    # SceneWorks' own trainer stamps networkType=lokr (epic 2193).
    _patch_adapter(monkeypatch, metadata={"networkType": "lokr"})
    assert classify_adapter_network("x.safetensors") == "lokr"


def test_kohya_lycoris_detected_by_metadata(monkeypatch):
    _patch_adapter(monkeypatch, metadata={"ss_network_module": "lycoris.kohya"})
    assert classify_adapter_network("x.safetensors") == "lycoris"


def test_lycoris_detected_by_lokr_keys_without_metadata(monkeypatch):
    # The real failing case: no networkType, no ss_network_module — detect by keys.
    _patch_adapter(
        monkeypatch,
        metadata={},
        tensor_names=[
            "lora_unet_transformer_blocks_0_attn_to_q.alpha",
            "lora_unet_transformer_blocks_0_attn_to_q.lokr_w1",
            "lora_unet_transformer_blocks_0_attn_to_q.lokr_w2",
        ],
    )
    assert classify_adapter_network("x.safetensors") == "lycoris"


def test_loha_keys_detected_as_lycoris(monkeypatch):
    _patch_adapter(monkeypatch, metadata={}, tensor_names=["x.hada_w1_a", "x.hada_w2_a"])
    assert classify_adapter_network("x.safetensors") == "lycoris"


def test_plain_lora_is_lora(monkeypatch):
    _patch_adapter(
        monkeypatch,
        metadata={},
        tensor_names=[
            "transformer.blocks.0.attn.to_q.lora_A.weight",
            "transformer.blocks.0.attn.to_q.lora_B.weight",
        ],
    )
    assert classify_adapter_network("x.safetensors") == "lora"


def test_reject_lokr_loras_rejects_lycoris(monkeypatch):
    monkeypatch.setattr(lora_adapters, "classify_adapter_network", lambda _p: "lycoris")
    spec = LoraSpec(id="snof", path="x.safetensors", weight=1.0, adapter_name="snof")
    with pytest.raises(RuntimeError) as excinfo:
        lora_adapters.reject_lokr_loras([spec], "mlx_qwen")
    msg = str(excinfo.value)
    assert "LyCORIS" in msg
    assert "snof" in msg


def test_reject_lokr_loras_allows_plain_lora(monkeypatch):
    monkeypatch.setattr(lora_adapters, "classify_adapter_network", lambda _p: "lora")
    spec = LoraSpec(id="ok", path="x.safetensors", weight=1.0, adapter_name="ok")
    lora_adapters.reject_lokr_loras([spec], "mlx_qwen")  # must not raise


# --------------------------------------------------------------------------- #
# torch + lycoris round-trip through the production path
# --------------------------------------------------------------------------- #


def test_lycoris_apply_and_restore_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lycoris")
    import torch.nn as nn
    from safetensors.torch import save_file

    class Node(nn.Module):
        pass

    # A tiny DiT-ish denoiser: transformer_blocks.0.attn.{to_q,to_k} as Linears.
    transformer = Node()
    blocks = Node()
    transformer.add_module("transformer_blocks", blocks)
    b0 = Node()
    blocks.add_module("0", b0)
    attn = Node()
    b0.add_module("attn", attn)
    attn.add_module("to_q", nn.Linear(16, 16, bias=False))
    attn.add_module("to_k", nn.Linear(16, 16, bias=False))

    # Hand-craft a kohya-format LoKr state dict (lokr_w1 @ lokr_w2 = a full 16x16
    # delta) for both leaves. factor split 4*4 = 16.
    def lokr_tensors():
        return {
            "lokr_w1": torch.randn(4, 4) * 0.3,
            "lokr_w2": torch.randn(4, 4) * 0.3,
            "alpha": torch.tensor(4.0),
        }

    state = {}
    for leaf in ("to_q", "to_k"):
        name = f"lora_unet_transformer_blocks_0_attn_{leaf}"
        for k, v in lokr_tensors().items():
            state[f"{name}.{k}"] = v
    path = tmp_path / "kohya_lokr.safetensors"
    save_file(state, str(path), metadata={"ss_network_module": "lycoris.kohya"})

    # Sanity: the production classifier sees this file as lycoris.
    assert classify_adapter_network(str(path)) == "lycoris"

    class FakePipe:
        def __init__(self, tr):
            self.transformer = tr

        def load_lora_weights(self, *a, **k):  # must never be called for LyCORIS
            raise AssertionError("load_lora_weights called for a LyCORIS adapter")

    pipe = FakePipe(transformer)
    to_q = transformer.transformer_blocks._modules["0"].attn.to_q
    torch.manual_seed(0)
    x = torch.randn(2, 16)
    base = to_q(x).clone()

    lora = {"id": "snof", "path": str(path), "weight": 1.0, "name": "snof"}
    pstate = lora_adapters.apply_loras_to_pipeline(
        pipe, [lora], adapter_id="qwen_image", model_family="qwen"
    )
    applied = to_q(x)
    assert (applied - base).abs().max().item() > 1e-6, "LyCORIS adapter did not change output"

    lora_adapters.clear_loras(pipe, pstate.adapter_names, adapter_id="qwen_image")
    restored = to_q(x)
    assert (restored - base).abs().max().item() < 1e-6, "clear_loras left LyCORIS residue"

    # Re-apply after clear proves cached-pipeline reuse isn't corrupted.
    pstate2 = lora_adapters.apply_loras_to_pipeline(
        pipe, [lora], adapter_id="qwen_image", model_family="qwen"
    )
    assert (to_q(x) - base).abs().max().item() > 1e-6
    lora_adapters.clear_loras(pipe, pstate2.adapter_names, adapter_id="qwen_image")
