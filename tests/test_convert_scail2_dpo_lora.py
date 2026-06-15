from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class FakeTensor:
    shape = (1, 1)

    def to(self, _dtype):
        return self

    def contiguous(self):
        return self


def load_converter(monkeypatch, saved):
    def fake_load(*args, **kwargs):
        saved["load"] = (args, kwargs)
        return {
            "module": {
                "model.diffusion_model.transformer.layers.0.attention.dense.lora_layer.down.weight": FakeTensor(),
                "model.diffusion_model.transformer.layers.0.attention.dense.lora_layer.up.weight": FakeTensor(),
            }
        }

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        load=fake_load,
    )
    fake_safetensors_torch = types.SimpleNamespace(
        save_file=lambda out, path: saved.setdefault("save", (out, path))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "safetensors", types.ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_scail2_dpo_lora.py"
    spec = importlib.util.spec_from_file_location("convert_scail2_dpo_lora_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_scail2_dpo_lora_uses_restricted_torch_load(monkeypatch):
    saved = {}
    converter = load_converter(monkeypatch, saved)

    converter.convert("input.pt", "out.safetensors")

    _args, kwargs = saved["load"]
    assert kwargs["map_location"] == "cpu"
    assert kwargs["weights_only"] is True
    out, path = saved["save"]
    assert path == "out.safetensors"
    assert sorted(out) == [
        "blocks.0.self_attn.o.lora_down.weight",
        "blocks.0.self_attn.o.lora_up.weight",
    ]
