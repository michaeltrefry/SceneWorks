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
# Prefix derivation (no torch / no lycoris).
#
# The lycoris loader addresses a denoiser submodule as ``f"{PREFIX}_{name}"`` (the
# module path with dots → underscores). Exporters disagree on PREFIX, and picking
# the wrong one matches 0 modules and aborts the apply. Recover it from the file's
# keys against the model's real submodule names (epic 2193).
# --------------------------------------------------------------------------- #


class _FakeNamedModules:
    """Minimal stand-in exposing ``named_modules()`` for prefix derivation."""

    def __init__(self, names):
        self._names = list(names)

    def named_modules(self):
        # The root module yields a "" name; include it like a real nn.Module does.
        return [("", object())] + [(name, object()) for name in self._names]


_QWEN_MODULE_NAMES = (
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.add_k_proj",
    "transformer_blocks.0.attn.to_add_out",
    "transformer_blocks.0.img_mlp.net.0.proj",
)


def _patch_tensor_names(monkeypatch, names):
    monkeypatch.setattr(lora_adapters, "_adapter_tensor_names", lambda _p, limit=256: list(names))


def test_prefix_derived_lycoris_library_default(monkeypatch):
    # The real failing file: keys carry the lycoris-lora library's own default
    # `lycoris` prefix, which the old known-prefix sniff defaulted past to lora_unet.
    _patch_tensor_names(
        monkeypatch,
        [f"lycoris_{name.replace('.', '_')}.lokr_w1" for name in _QWEN_MODULE_NAMES],
    )
    module = _FakeNamedModules(_QWEN_MODULE_NAMES)
    assert lora_adapters._lycoris_module_prefix("x.safetensors", module) == "lycoris"


def test_prefix_derived_kohya_lora_unet(monkeypatch):
    _patch_tensor_names(
        monkeypatch,
        [f"lora_unet_{name.replace('.', '_')}.lokr_w1" for name in _QWEN_MODULE_NAMES],
    )
    module = _FakeNamedModules(_QWEN_MODULE_NAMES)
    assert lora_adapters._lycoris_module_prefix("x.safetensors", module) == "lora_unet"


def test_prefix_sniff_recognizes_lycoris_without_module(monkeypatch):
    # No module to match against → fall back to the known-prefix sniff, which must
    # still recognize the `lycoris` default (not blindly return lora_unet).
    _patch_tensor_names(monkeypatch, ["lycoris_transformer_blocks_0_attn_to_q.lokr_w1"])
    assert lora_adapters._lycoris_module_prefix("x.safetensors") == "lycoris"


def test_prefix_falls_back_to_lora_unet_on_no_match(monkeypatch):
    # Keys whose module path matches nothing in the model (wrong architecture) → no
    # vote → sniff → default lora_unet, so the apply raises the clear 0-match error.
    _patch_tensor_names(monkeypatch, ["weird_blocks_0_thing.lokr_w1"])
    module = _FakeNamedModules(_QWEN_MODULE_NAMES)
    assert lora_adapters._lycoris_module_prefix("x.safetensors", module) == "lora_unet"


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


def test_lycoris_library_default_prefix_roundtrip(tmp_path):
    """A LoKr exported by the lycoris-lora library (keys prefixed ``lycoris_``, its
    ``save_weights`` default) must apply on a DiT denoiser — the prefix is recovered
    from the model's submodule names, not assumed to be ``lora_unet`` (epic 2193)."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("lycoris")
    import json

    import torch.nn as nn
    from safetensors.torch import save_file

    class Node(nn.Module):
        pass

    transformer = Node()
    blocks = Node()
    transformer.add_module("transformer_blocks", blocks)
    b0 = Node()
    blocks.add_module("0", b0)
    attn = Node()
    b0.add_module("attn", attn)
    attn.add_module("to_q", nn.Linear(16, 16, bias=False))
    attn.add_module("to_k", nn.Linear(16, 16, bias=False))

    def lokr_tensors():
        return {
            "lokr_w1": torch.randn(4, 4) * 0.3,
            "lokr_w2": torch.randn(4, 4) * 0.3,
            "alpha": torch.tensor(4.0),
        }

    state = {}
    for leaf in ("to_q", "to_k"):
        # The lycoris-lora library's own prefix — the exact case the old sniff missed.
        name = f"lycoris_transformer_blocks_0_attn_{leaf}"
        for k, v in lokr_tensors().items():
            state[f"{name}.{k}"] = v
    path = tmp_path / "lycoris_lokr.safetensors"
    save_file(
        state,
        str(path),
        metadata={"lycoris_config": json.dumps({"algo": "lokr", "factor": 4})},
    )

    # No networkType / ss_network_module stamp — detected purely by lokr_* keys.
    assert classify_adapter_network(str(path)) == "lycoris"

    class FakePipe:
        def __init__(self, tr):
            self.transformer = tr

        def load_lora_weights(self, *a, **k):
            raise AssertionError("load_lora_weights called for a LyCORIS adapter")

    pipe = FakePipe(transformer)
    to_q = transformer.transformer_blocks._modules["0"].attn.to_q
    torch.manual_seed(0)
    x = torch.randn(2, 16)
    base = to_q(x).clone()

    lora = {"id": "human_focus", "path": str(path), "weight": 1.0, "name": "human_focus"}
    pstate = lora_adapters.apply_loras_to_pipeline(
        pipe, [lora], adapter_id="qwen_image", model_family="qwen"
    )
    assert (to_q(x) - base).abs().max().item() > 1e-6, "lycoris-prefixed LoKr did not apply"

    lora_adapters.clear_loras(pipe, pstate.adapter_names, adapter_id="qwen_image")
    assert (to_q(x) - base).abs().max().item() < 1e-6, "clear_loras left LyCORIS residue"
