from __future__ import annotations

from worker_runtime_shared import *

@pytest.fixture(autouse=True)
def managed_lora_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENEWORKS_DATA_DIR", str(tmp_path))


def test_cpu_worker_does_not_advertise_lora_train():
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})

    assert "lora_train" not in capabilities

def test_qwen_distill_lora_helpers():
    base_target = MODEL_TARGETS["qwen_image_edit_2511"]
    lightning_target = MODEL_TARGETS["qwen_image_edit_2511_lightning"]
    assert QwenImageAdapter._distill_lora_for(base_target) is None
    assert QwenImageAdapter._distill_key_for(None) is None
    spec = QwenImageAdapter._distill_lora_for(lightning_target)
    assert spec is not None
    key = QwenImageAdapter._distill_key_for(spec)
    assert key == "lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

def test_qwen_character_image_angle_set_applies_loras_once_then_loops_angles(tmp_path, monkeypatch):
    """sc-2225: a character_image + angleSet job on a diffusers backbone (Qwen) must
    apply request.loras exactly once, BEFORE the per-angle loop, and emit one image per
    canonical angle. Guards the regression that angle-set mode skips the LoRA merge."""

    class FakeImage:
        def convert(self, _mode):
            return self

    # The worker unit suite runs without torch installed; generate() does
    # `importlib.import_module("torch")`, so stand in a fake. With gpu_id="cpu",
    # select_torch_device returns early and never touches it.
    class FakeTorch:
        pass

    adapter = QwenImageAdapter()
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr("scene_worker.image_adapters.gpu_memory_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(adapter, "_load_pipeline", lambda *a, **k: object())

    apply_calls: list[list] = []
    monkeypatch.setattr(
        adapter, "_apply_loras", lambda pipe, request, lora_key=None: apply_calls.append(list(request.loras))
    )

    run_overrides: list = []

    # generate() always passes pose_skeleton (None for the angle path); accept it
    # here so the fake matches the real _run_pipeline signature (sc-2256).
    def fake_run(settings, pipe, request, seed, project_path, *, cancel_requested=None, prompt_override=None, pose_skeleton=None):
        # The LoRA merge must already have happened before any angle is generated.
        assert apply_calls, "loras must be applied before the angle loop runs"
        assert pose_skeleton is None  # angle path is prompt-driven, no skeleton
        run_overrides.append(prompt_override)
        return FakeImage()

    monkeypatch.setattr(adapter, "_run_pipeline", fake_run)

    captured: dict = {}

    def fake_writer(self, *, image_count, image_at_index, **kwargs):
        captured["image_count"] = image_count
        for index in range(image_count):
            image_at_index(index)
        return {"images": [], "count": image_count}

    monkeypatch.setattr(ImageAssetWriter, "write_incremental_outputs", fake_writer)

    loras = [{"id": "kelsie", "path": "/loras/kelsie.safetensors", "families": ["qwen-image"]}]
    job = {
        "id": "job-qwen-angle",
        "payload": {
            "projectId": "p",
            "mode": "character_image",
            "model": "qwen_image_edit_2511",
            "prompt": "the character",
            "referenceAssetId": "ref-1",
            "count": 1,
            "width": 16,
            "height": 16,
            "loras": loras,
            "advanced": {"angleSet": True},
        },
    }
    adapter.generate(
        settings=SimpleNamespace(gpu_id="cpu"),
        job=job,
        request=image_request_from_job(job),
        project_path=tmp_path,
        progress=lambda *a, **k: None,
        cancel_requested=lambda: False,
    )

    # Applied exactly once (per pipe, before the loop) with the requested loras.
    assert apply_calls == [loras]
    # One image per canonical angle, each with its own per-angle prompt augment.
    assert captured["image_count"] == len(CHARACTER_ANGLE_SET_ORDER)
    assert len(run_overrides) == len(CHARACTER_ANGLE_SET_ORDER)
    assert len(set(run_overrides)) == len(CHARACTER_ANGLE_SET_ORDER)

def test_filter_call_kwargs_preserves_none_for_accepted_parameters():
    assert filter_call_kwargs(AcceptsNone(), {"prompt": "city", "image": None, "extra": 1}) == {
        "prompt": "city",
        "image": None,
    }

def test_lora_loader_applies_weights_and_reuses_cached_state(tmp_path):
    first = tmp_path / "style.safetensors"
    second = tmp_path / "detail.safetensors"
    first.write_bytes(b"lora")
    second.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    loras = [
        {"id": "style", "installedPath": str(first), "weight": 0.5},
        {"id": "detail", "installedPath": str(second), "weight": 0.8},
    ]

    state = apply_loras_to_pipeline(pipe, loras, adapter_id="diffusers_test")
    same_state = apply_loras_to_pipeline(
        pipe,
        loras,
        adapter_id="diffusers_test",
        previous_state=state,
    )

    assert same_state == state
    assert [path for path, _name in pipe.loaded] == [str(first), str(second)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.5, 0.8])]
    assert pipe.unloaded == 0

def test_wan_moe_lora_loads_low_noise_into_transformer_2(tmp_path):
    # A trained Wan A14B MoE LoRA is a dir with a high/low pair (sc-1953). The
    # high-noise half loads into the transformer, the low-noise half into
    # transformer_2 via diffusers' load_into_transformer_2 (sc-1955).
    moe = tmp_path / "lora_moe"
    moe.mkdir()
    (moe / "char.high_noise.safetensors").write_bytes(b"lora")
    (moe / "char.low_noise.safetensors").write_bytes(b"lora")
    lora = {
        "id": "char",
        "installedPath": str(moe),
        "families": ["wan-video"],
        "baseModel": "wan_2_2_t2v_14b",
        "weight": 0.8,
    }

    specs = normalize_lora_specs([lora])
    assert specs[0].path.endswith("char.high_noise.safetensors")
    assert specs[0].secondary_path.endswith("char.low_noise.safetensors")

    pipe = FakeMoeLoraPipe()
    apply_loras_to_pipeline(
        pipe, [lora], adapter_id="wan_video", model_family="wan-video", model_id="wan_2_2_t2v_14b"
    )
    assert len(pipe.loaded) == 2
    high_path, high_name, high_t2 = pipe.loaded[0]
    low_path, low_name, low_t2 = pipe.loaded[1]
    assert high_path.endswith("char.high_noise.safetensors") and high_t2 is False
    assert low_path.endswith("char.low_noise.safetensors") and low_t2 is True
    # Both experts share the same adapter name so set_adapters activates both.
    assert high_name == low_name

def test_wan_moe_lora_on_dense_pipe_skips_second_expert(tmp_path):
    # A MoE LoRA on a pipe without transformer_2 loads only the high-noise half
    # (base-model gating blocks this combo upstream, but it must not crash).
    moe = tmp_path / "lora_moe2"
    moe.mkdir()
    (moe / "char.high_noise.safetensors").write_bytes(b"lora")
    (moe / "char.low_noise.safetensors").write_bytes(b"lora")
    pipe = FakeLoraPipe()  # dense: no transformer_2
    apply_loras_to_pipeline(
        pipe,
        [{"id": "char", "installedPath": str(moe), "families": ["wan-video"]}],
        adapter_id="wan_video",
        model_family="wan-video",
        model_id="wan_2_2",
    )
    assert len(pipe.loaded) == 1
    assert pipe.loaded[0][0].endswith("char.high_noise.safetensors")

def test_validate_lora_compatibility_gates_wan_base_model():
    wan_5b = {"id": "l", "families": ["wan-video"], "baseModel": "wan_2_2"}
    # A 5B LoRA on a 14B model is rejected (both family wan-video, incompatible arch).
    with pytest.raises(RuntimeError, match="not interchangeable"):
        validate_lora_compatibility(
            [wan_5b], model_family="wan-video", adapter_id="wan_video", model_id="wan_2_2_t2v_14b"
        )
    # Exact base-model match passes.
    validate_lora_compatibility(
        [wan_5b], model_family="wan-video", adapter_id="wan_video", model_id="wan_2_2"
    )
    # No recorded baseModel -> family gating only (legacy/imported), no rejection.
    validate_lora_compatibility(
        [{"id": "l2", "families": ["wan-video"]}],
        model_family="wan-video",
        adapter_id="wan_video",
        model_id="wan_2_2_t2v_14b",
    )
    # No model_id -> base-model gate is inert (back-compat with other call sites).
    validate_lora_compatibility([wan_5b], model_family="wan-video", adapter_id="wan_video")

def test_lora_loader_clears_previous_adapters_between_jobs(tmp_path):
    first = tmp_path / "style.safetensors"
    second = tmp_path / "detail.safetensors"
    first.write_bytes(b"lora")
    second.write_bytes(b"lora")
    pipe = FakeLoraPipe()

    state = apply_loras_to_pipeline(pipe, [{"id": "style", "installedPath": str(first)}], adapter_id="diffusers_test")
    apply_loras_to_pipeline(
        pipe,
        [{"id": "detail", "installedPath": str(second)}],
        adapter_id="diffusers_test",
        previous_state=state,
    )

    assert pipe.unloaded == 1
    assert pipe.loaded[-1][0] == str(second)

def test_adapter_network_type_defaults_to_lora_for_plain_file(tmp_path):
    plain = tmp_path / "plain.safetensors"
    plain.write_bytes(b"not a real safetensors header")
    # Unreadable/absent metadata resolves to lora — every legacy adapter is lora.
    assert adapter_network_type(plain) == "lora"

def test_reject_lokr_loras_raises_only_for_lokr(monkeypatch, tmp_path):
    lora_file = tmp_path / "a.safetensors"
    lora_file.write_bytes(b"x")
    lokr_file = tmp_path / "b.safetensors"
    lokr_file.write_bytes(b"x")
    types = {str(lora_file): "lora", str(lokr_file): "lokr"}
    monkeypatch.setattr(
        "scene_worker.lora_adapters.adapter_network_type", lambda path: types[str(path)]
    )
    lora_spec = LoraSpec(id="a", path=str(lora_file), weight=1.0, adapter_name="a")
    lokr_spec = LoraSpec(id="b", path=str(lokr_file), weight=1.0, adapter_name="b")

    reject_lokr_loras([lora_spec], "mlx_test")  # all-lora: no raise
    with pytest.raises(RuntimeError, match="LoKr"):
        reject_lokr_loras([lora_spec, lokr_spec], "mlx_test")

def test_apply_loras_routes_lokr_through_injection(monkeypatch, tmp_path):
    lokr_file = tmp_path / "char.safetensors"
    lokr_file.write_bytes(b"x")
    monkeypatch.setattr("scene_worker.lora_adapters.adapter_network_type", lambda path: "lokr")
    injected = []
    monkeypatch.setattr(
        "scene_worker.lora_adapters.inject_lokr_adapter",
        lambda pipe, spec, *, adapter_id: injected.append(spec.adapter_name),
    )
    pipe = FakeLokrPipe()

    state = apply_loras_to_pipeline(
        pipe,
        [{"id": "char", "installedPath": str(lokr_file), "weight": 0.7}],
        adapter_id="sdxl_test",
    )

    # LoKr never calls load_lora_weights; it injects, then sets weight on the module.
    assert pipe.loaded == []
    assert injected == list(state.adapter_names)
    assert len(state.adapter_names) == 1
    assert pipe.unet.set_calls[-1] == (list(state.adapter_names), [0.7])

def test_clear_loras_prefers_delete_adapters_so_lokr_is_removed():
    # delete_adapters removes injected LoKr adapters; unload_lora_weights (LoRA-only)
    # would leak them into the next job.
    pipe = FakeTargetedLoraPipe()
    clear_loras(pipe, ("char",), adapter_id="sdxl_test")
    assert pipe.deleted == [["char"]]
    assert pipe.unloaded == 0

def test_clear_loras_mixed_lycoris_and_peft_deletes_only_peft_names(monkeypatch):
    # sc-4181: on a cached pipeline holding a LyCORIS net AND a peft adapter,
    # clear_loras must pass only the non-LyCORIS leftovers to delete_adapters —
    # the full list includes the just-restored LyCORIS name, which diffusers'
    # delete_adapters rejects, failing the next job.
    monkeypatch.setattr(
        "scene_worker.lora_adapters._restore_lycoris_nets",
        lambda module, names: {"lyc_style"},
    )
    pipe = FakeTargetedLoraPipe()
    clear_loras(pipe, ("lyc_style", "char"), adapter_id="sdxl_test")
    assert pipe.deleted == [["char"]]
    assert pipe.unloaded == 0

def test_set_adapter_weights_on_module_applies_and_guards():
    module = FakeDenoiserModule()
    spec = LoraSpec(id="c", path="p", weight=0.5, adapter_name="c")
    set_adapter_weights_on_module(module, ("c",), [0.5], adapter_id="t", specs=[spec])
    assert module.set_calls == [(["c"], [0.5])]

    # No module support: a single full-weight adapter is already active (fine),
    # but multiple / non-unity weights cannot be honored.
    set_adapter_weights_on_module(None, ("c",), [1.0], adapter_id="t", specs=[spec])
    with pytest.raises(RuntimeError, match="per-adapter weights"):
        set_adapter_weights_on_module(None, ("c", "d"), [1.0, 0.5], adapter_id="t", specs=[spec])

def test_lora_loader_reuses_overlap_when_adapter_can_delete_targeted_loras(tmp_path):
    first = tmp_path / "style.safetensors"
    second = tmp_path / "detail.safetensors"
    third = tmp_path / "motion.safetensors"
    first.write_bytes(b"lora")
    second.write_bytes(b"lora")
    third.write_bytes(b"lora")
    pipe = FakeTargetedLoraPipe()
    state = apply_loras_to_pipeline(
        pipe,
        [{"id": "style", "installedPath": str(first)}, {"id": "detail", "installedPath": str(second)}],
        adapter_id="diffusers_test",
    )

    apply_loras_to_pipeline(
        pipe,
        [{"id": "style", "installedPath": str(first)}, {"id": "motion", "installedPath": str(third)}],
        adapter_id="diffusers_test",
        previous_state=state,
    )

    assert pipe.unloaded == 0
    assert pipe.deleted == [[state.specs[1].adapter_name]]
    assert [path for path, _name in pipe.loaded] == [str(first), str(second), str(third)]

def test_lora_cache_key_is_stable_for_reordered_loras(tmp_path):
    first = tmp_path / "style.safetensors"
    second = tmp_path / "detail.safetensors"
    first.write_bytes(b"lora")
    second.write_bytes(b"lora")
    left = [{"id": "style", "installedPath": str(first), "weight": 0.5}, {"id": "detail", "installedPath": str(second)}]
    right = list(reversed(left))

    key = lora_cache_key(left)
    assert key == lora_cache_key(right)
    assert len(key) == 64

def test_first_safetensors_path_prefers_final_over_step_checkpoints(tmp_path):
    # A trained-LoRA directory holds the final adapter plus per-step checkpoints.
    for step in (250, 500, 3000):
        (tmp_path / f"kelsie_lora-step{step:06d}.safetensors").write_bytes(b"ckpt")
    final = tmp_path / "kelsie_lora.safetensors"
    final.write_bytes(b"final")

    assert first_safetensors_path(tmp_path) == final

def test_first_safetensors_path_picks_latest_checkpoint_when_no_final(tmp_path):
    for step in (250, 500, 3000):
        (tmp_path / f"kelsie_lora-step{step:06d}.safetensors").write_bytes(b"ckpt")

    assert first_safetensors_path(tmp_path) == tmp_path / "kelsie_lora-step003000.safetensors"

def test_resolve_lora_file_uses_declared_files_over_checkpoints(tmp_path):
    (tmp_path / "kelsie_lora-step000250.safetensors").write_bytes(b"ckpt")
    final = tmp_path / "kelsie_lora.safetensors"
    final.write_bytes(b"final")

    resolved = resolve_lora_file(tmp_path, {"files": ["kelsie_lora.safetensors"]})

    assert resolved == final

def test_lora_loader_allows_single_implicit_weight_without_set_adapters(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")
    pipe = FakeSingleLoraPipe()

    state = apply_loras_to_pipeline(pipe, [{"id": "style", "installedPath": str(first), "weight": 1.0}], adapter_id="diffusers_test")

    assert state.adapter_names
    assert pipe.loaded[0][0] == str(first)

def test_lora_loader_fails_when_pipeline_cannot_load_loras(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")

    with pytest.raises(RuntimeError, match="does not support loading LoRA weights"):
        apply_loras_to_pipeline(object(), [{"id": "style", "installedPath": str(first)}], adapter_id="diffusers_test")

def test_lora_loader_explains_missing_peft_backend(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")

    with pytest.raises(RuntimeError, match="LoRA style requires the PEFT backend") as info:
        apply_loras_to_pipeline(
            FakePeftBackendErrorPipe(),
            [{"id": "style", "installedPath": str(first)}],
            adapter_id="diffusers_test",
        )

    assert isinstance(info.value.__cause__, ValueError)
    assert "docker compose build worker --no-cache" in str(info.value)

def test_lora_loader_detects_reworded_peft_backend_errors(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")

    class MissingPeftPipe:
        def load_lora_weights(self, path, adapter_name=None):
            raise ModuleNotFoundError("No module named 'peft'")

    with pytest.raises(RuntimeError, match="LoRA style requires the PEFT backend"):
        apply_loras_to_pipeline(
            MissingPeftPipe(),
            [{"id": "style", "installedPath": str(first)}],
            adapter_id="diffusers_test",
        )

def test_unsupported_adapter_guard_rejects_loras(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")

    with pytest.raises(RuntimeError, match="does not support LoRA application"):
        reject_loras_if_unsupported([{"id": "style", "installedPath": str(first)}], "procedural_preview")

def test_lora_compatibility_guard_rejects_mismatched_family_before_load(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")
    pipe = FakeLoraPipe()

    with pytest.raises(RuntimeError, match="LoRA style is not compatible with model family z-image"):
        apply_loras_to_pipeline(
            pipe,
            [{"id": "style", "installedPath": str(first), "family": "qwen_image"}],
            adapter_id="diffusers_test",
            model_family="z-image",
        )

    assert pipe.loaded == []

def test_lora_compatibility_guard_soft_passes_legacy_jobs_without_family(tmp_path):
    first = tmp_path / "style.safetensors"
    first.write_bytes(b"lora")
    pipe = FakeLoraPipe()

    apply_loras_to_pipeline(
        pipe,
        [{"id": "style", "installedPath": str(first)}],
        adapter_id="diffusers_test",
        model_family="z-image",
    )

    assert pipe.loaded[0][0] == str(first)

def test_lora_compatibility_guard_accepts_normalized_family_aliases():
    validate_lora_compatibility(
        [{"id": "style", "compatibility": {"families": ["z_image"]}}],
        model_family="z-image",
        adapter_id="diffusers_test",
    )

def test_lora_compatibility_chroma_accepts_flux_but_not_vice_versa():
    # Chroma accepts flux LoRAs (FLUX.1-schnell-derived, identical keys). sc-1832.
    validate_lora_compatibility(
        [{"id": "flux_style", "compatibility": {"families": ["flux"]}}],
        model_family="chroma",
        adapter_id="diffusers_test",
    )
    # ...and chroma-tagged LoRAs.
    validate_lora_compatibility(
        [{"id": "chroma_style", "compatibility": {"families": ["chroma"]}}],
        model_family="chroma",
        adapter_id="diffusers_test",
    )
    # The relationship is one-directional: a Flux model rejects a chroma LoRA.
    with pytest.raises(RuntimeError, match="not compatible with model family flux"):
        validate_lora_compatibility(
            [{"id": "chroma_style", "compatibility": {"families": ["chroma"]}}],
            model_family="flux",
            adapter_id="diffusers_test",
        )

def test_lora_compatibility_flux2_klein_accepts_flux2_lora():
    # FLUX.2 [klein] models have family "flux2-klein" but accept "flux2" LoRAs
    # (loraCompatibility.families = ["flux2"]). The validate guard keys off the
    # model `family` string, so the klein->flux2 relationship must be declared.
    validate_lora_compatibility(
        [{"id": "portrait_engine", "compatibility": {"families": ["flux2"]}}],
        model_family="flux2-klein",
        adapter_id="mlx_flux2",
    )
    # The relationship is one-directional: a plain flux2 model would reject a
    # flux2-klein-tagged LoRA (no such model ships today, but keep the gate tight).
    with pytest.raises(RuntimeError, match="not compatible with model family flux2"):
        validate_lora_compatibility(
            [{"id": "klein_only", "compatibility": {"families": ["flux2-klein"]}}],
            model_family="flux2",
            adapter_id="mlx_flux2",
        )

def test_lora_weight_defaults_on_unparseable_values():
    assert lora_weight({"weight": "not-a-number"}) == 0.8

def test_lora_specs_fail_before_inference_for_missing_or_excess_loras(tmp_path):
    missing = tmp_path / "missing.safetensors"

    with pytest.raises(RuntimeError, match="file is missing"):
        normalize_lora_specs([{"id": "missing", "installedPath": str(missing)}])

    empty_dir = tmp_path / "empty_lora"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match=r"LoRA empty has no \.safetensors file"):
        normalize_lora_specs([{"id": "empty", "installedPath": str(empty_dir)}])

    many = [{"id": f"lora_{index}", "installedPath": str(missing)} for index in range(6)]
    with pytest.raises(RuntimeError, match="at most 5 LoRAs"):
        normalize_lora_specs(many)


def test_lora_specs_reject_outside_app_managed_root(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "evil.safetensors"
    outside.parent.mkdir()
    outside.write_bytes(b"lora")

    with pytest.raises(RuntimeError, match="inside an app-managed directory"):
        normalize_lora_specs([{"id": "evil", "installedPath": str(outside)}])


def test_lora_specs_resolve_installed_directory_to_safetensors_file(tmp_path):
    # Installed LoRAs are stored as a directory and the Rust API reports the
    # directory as `installedPath`. The native ltx-core loader mmaps the path
    # directly, and mmap on a directory raises ENODEV ("No such device (os error
    # 19)"), so the spec path must point at the .safetensors file, not the dir.
    lora_dir = tmp_path / "loras" / "lauren"
    lora_dir.mkdir(parents=True)
    weights = lora_dir / "Lauren_ltx2.3.safetensors"
    weights.write_bytes(b"")

    specs = normalize_lora_specs(
        [
            {
                "id": "lauren",
                "weight": 0.8,
                "files": ["Lauren_ltx2.3.safetensors"],
                "installedPath": str(lora_dir),
                "source": {"provider": "local", "path": "loras/lauren"},
            }
        ]
    )

    assert specs[0].path == str(weights)

def test_lora_specs_resolve_huggingface_cache_snapshot(monkeypatch, tmp_path):
    cache_root = tmp_path / "hf" / "hub"
    snapshot = write_huggingface_cache_resource(
        cache_root,
        "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    )
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))

    specs = normalize_lora_specs(
        [
            {
                "id": "ltx_2_3_ic_union_control",
                "weight": 0.7,
                "source": {
                    "provider": "huggingface",
                    "repo": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                    "file": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                },
            }
        ]
    )

    assert specs[0].path == str(snapshot / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors")
    assert specs[0].weight == 0.7

def test_lora_specs_prefer_huggingface_ref_main_snapshot(monkeypatch, tmp_path):
    cache_root = tmp_path / "hf" / "hub"
    repo = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
    file_name = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    write_huggingface_cache_resource(cache_root, repo, file_name, revision="aaa111")
    main_snapshot = write_huggingface_cache_resource(cache_root, repo, file_name, revision="zzz999", refs_main=True)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))

    specs = normalize_lora_specs(
        [
            {
                "id": "ltx_2_3_ic_union_control",
                "source": {
                    "provider": "huggingface",
                    "repo": repo,
                    "file": file_name,
                },
            }
        ]
    )

    assert specs[0].path == str(main_snapshot / file_name)

def test_lens_save_lora_routes_by_network_type(monkeypatch, tmp_path):
    from scene_worker import lens_train_runner as lt

    # Plain LoRA: delegates to the transformer's diffusers PeftAdapterMixin saver.
    saved: dict[str, object] = {}

    class FakeTransformer:
        def save_lora_adapter(self, output_dir, *, weight_name=None, safe_serialization=None):
            saved["output_dir"] = output_dir
            saved["weight_name"] = weight_name

    out = lt._save_lora(FakeTransformer(), str(tmp_path), "lens.safetensors", network_type="lora")
    assert saved["weight_name"] == "lens.safetensors"
    assert out == str(tmp_path / "lens.safetensors")

    # LoKr: routes to the raw metadata writer instead (lokr_w1/w2 aren't
    # save_lora_adapter-compatible), and never calls save_lora_adapter.
    captured: dict[str, object] = {}

    def fake_save_lokr(transformer, output_dir, file_name, **kwargs):
        captured["file_name"] = file_name
        captured["kwargs"] = kwargs
        return str(tmp_path / file_name)

    monkeypatch.setattr(lt, "_save_lokr_adapter", fake_save_lokr)

    class ExplodingTransformer:
        def save_lora_adapter(self, *a, **k):
            raise AssertionError("LoKr must not save via save_lora_adapter")

    out = lt._save_lora(
        ExplodingTransformer(), str(tmp_path), "lens.safetensors",
        network_type="lokr", rank=8, alpha=8, decompose_factor=4,
        target_modules=["img_qkv", "txt_qkv"],
    )
    assert captured["file_name"] == "lens.safetensors"
    assert captured["kwargs"] == {
        "rank": 8, "alpha": 8, "decompose_factor": 4,
        "target_modules": ["img_qkv", "txt_qkv"],
    }

def test_lens_save_lokr_adapter_stamps_metadata(monkeypatch, tmp_path):
    import sys
    import types as types_module

    from scene_worker import lens_train_runner as lt

    fake_state = {"transformer.x.lokr_w1": "t1", "transformer.x.lokr_w2": "t2"}
    fake_peft_utils = types_module.ModuleType("peft.utils")
    fake_peft_utils.get_peft_model_state_dict = lambda _m: fake_state
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)

    written: dict[str, object] = {}

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

    fake_st_torch = types_module.ModuleType("safetensors.torch")

    def fake_save_file(tensors, path, metadata=None):
        written["tensors"] = dict(tensors)
        written["metadata"] = metadata

    fake_st_torch.save_file = fake_save_file
    monkeypatch.setitem(sys.modules, "safetensors", types_module.ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_st_torch)

    # get_peft_model_state_dict returns our fake tensors; patch their .detach chain.
    monkeypatch.setattr(fake_peft_utils, "get_peft_model_state_dict", lambda _m: {
        "transformer.x.lokr_w1": FakeTensor(), "transformer.x.lokr_w2": FakeTensor(),
    })

    out = lt._save_lokr_adapter(
        object(), str(tmp_path), "lens_lokr.safetensors",
        rank=8, alpha=16, decompose_factor=4, target_modules=["img_qkv", "txt_qkv"],
    )
    assert out == str(tmp_path / "lens_lokr.safetensors")
    meta = written["metadata"]
    assert meta["networkType"] == "lokr"
    assert meta["rank"] == "8" and meta["alpha"] == "16" and meta["decomposeFactor"] == "4"
    assert json.loads(meta["targetModules"]) == ["img_qkv", "txt_qkv"]
    assert set(written["tensors"]) == {"transformer.x.lokr_w1", "transformer.x.lokr_w2"}

def test_lens_resolution_for_snaps_to_buckets():
    # Square requests pick the base by area: <1024*1440 px -> 1024, else 1440.
    assert lens_resolution_for(1024, 1024) == (1024, "1:1")
    assert lens_resolution_for(1440, 1440) == (1440, "1:1")
    assert lens_resolution_for(2048, 2048) == (1440, "1:1")
    # Aspect ratio snaps by closest log-ratio (W:H).
    assert lens_resolution_for(1280, 720) == (1024, "16:9")
    assert lens_resolution_for(720, 1280) == (1024, "9:16")
    assert lens_resolution_for(1152, 864) == (1024, "4:3")
    assert lens_resolution_for(864, 1152) == (1024, "3:4")

def test_request_has_lokr_lora_detection():
    from scene_worker.image_adapters import _request_has_lokr_lora

    # Recorded networkType (top-level, mirroring baseModel) or nested in
    # compatibility — both are read with no file I/O (epic 2193).
    assert _request_has_lokr_lora({"loras": [{"id": "a", "networkType": "lokr"}]}) is True
    assert _request_has_lokr_lora({"loras": [{"id": "a", "compatibility": {"networkType": "LoKr"}}]}) is True
    assert _request_has_lokr_lora({"loras": [{"id": "a", "networkType": "lora"}]}) is False
    assert _request_has_lokr_lora({"loras": []}) is False
    assert _request_has_lokr_lora({}) is False

def test_flux_adapter_applies_flux_lora(tmp_path):
    lora = tmp_path / "flux_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = FluxDiffusersAdapter()
    request = SimpleNamespace(
        model="flux_schnell",
        loras=[
            {
                "id": "flux_style",
                "installedPath": str(lora),
                "weight": 0.7,
                "compatibility": {"families": ["flux"]},
            }
        ],
    )
    adapter._apply_loras(pipe, request)
    state = adapter._loaded_lora_states["text"]
    assert [path for path, _name in pipe.loaded] == [str(lora)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.7])]

def test_flux_adapter_rejects_incompatible_lora_family(tmp_path):
    lora = tmp_path / "qwen_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = FluxDiffusersAdapter()
    request = SimpleNamespace(
        model="flux_dev",
        loras=[
            {
                "id": "qwen_style",
                "installedPath": str(lora),
                "compatibility": {"families": ["qwen-image"]},
            }
        ],
    )
    try:
        adapter._apply_loras(pipe, request)
    except RuntimeError as exc:
        assert "not compatible with model family flux" in str(exc)
    else:
        raise AssertionError("FLUX.1 must reject a LoRA whose family is not flux.")

def test_chroma_adapter_applies_chroma_lora(tmp_path):
    lora = tmp_path / "chroma_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = ChromaDiffusersAdapter()
    request = SimpleNamespace(
        model="chroma1_hd",
        loras=[
            {
                "id": "chroma_style",
                "installedPath": str(lora),
                "weight": 0.7,
                "compatibility": {"families": ["chroma"]},
            }
        ],
    )
    adapter._apply_loras(pipe, request)
    state = adapter._loaded_lora_states["text"]
    assert [path for path, _name in pipe.loaded] == [str(lora)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.7])]

def test_chroma_adapter_applies_flux_lora(tmp_path):
    # Chroma is FLUX.1-schnell-derived: Flux LoRAs (and Chroma LoRAs detected as
    # flux by their identical tensor keys) load on Chroma. sc-1832.
    lora = tmp_path / "flux_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = ChromaDiffusersAdapter()
    request = SimpleNamespace(
        model="chroma1_hd",
        loras=[
            {
                "id": "flux_style",
                "installedPath": str(lora),
                "weight": 0.8,
                "compatibility": {"families": ["flux"]},
            }
        ],
    )
    adapter._apply_loras(pipe, request)
    state = adapter._loaded_lora_states["text"]
    assert [path for path, _name in pipe.loaded] == [str(lora)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.8])]

def test_chroma_adapter_rejects_incompatible_lora_family(tmp_path):
    lora = tmp_path / "qwen_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = ChromaDiffusersAdapter()
    request = SimpleNamespace(
        model="chroma1_base",
        loras=[
            {
                "id": "qwen_style",
                "installedPath": str(lora),
                "compatibility": {"families": ["qwen-image"]},
            }
        ],
    )
    try:
        adapter._apply_loras(pipe, request)
    except RuntimeError as exc:
        assert "not compatible with model family chroma" in str(exc)
    else:
        raise AssertionError("Chroma1 must reject a LoRA whose family is neither chroma nor flux.")

def test_kolors_pose_set_applies_loras_once_before_loop(tmp_path, monkeypatch):
    """sc-2251/sc-2252: the strict Kolors pose tier must apply request.loras exactly
    once on the pose pipe BEFORE the per-pose loop (the same merge the T2I/img2img
    path does), under its own "pose" cache slot — so the character-LoRA bootstrapping
    loop works on pose generation. Guards the regression where _generate_pose_set
    returned before the LoRA merge, silently ignoring request.loras (and LoKr)."""
    from scene_worker import image_adapters as ia

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_load_pose_pipeline", lambda self, *a, **k: object())
    monkeypatch.setattr(ia, "select_torch_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: SimpleNamespace() if name == "torch" else importlib.import_module(name),
    )

    apply_calls: list[dict] = []

    def fake_apply_loras(self, pipe, request, *, lora_key=None):
        apply_calls.append({"loras": list(request.loras), "lora_key": lora_key})

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_apply_loras", fake_apply_loras)

    def fake_run_pose(self, settings, pipe, request, seed, project_path, keypoints, hands=None, face=None, cancel_requested=None):
        from PIL import Image as _Image

        assert apply_calls, "loras must be applied before the pose loop runs"
        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_run_pose", fake_run_pose)

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, **kwargs):
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)

    kp = [[0.5, 0.1 + 0.04 * i] for i in range(18)]
    loras = [{"id": "kelsie", "path": "/loras/kelsie.safetensors", "families": ["kolors"]}]
    job = {"id": "job_kolors_pose_lora", "payload": {
        "projectId": "p", "mode": "character_image", "model": "kolors", "prompt": "the character",
        "referenceAssetId": "ref-1", "count": 1, "width": 64, "height": 64, "loras": loras,
        "advanced": {"poses": [{"id": "sit_01", "keypoints": kp}, {"id": "stand_01", "keypoints": kp}]},
    }}
    KolorsDiffusersAdapter().generate(
        settings=SimpleNamespace(gpu_id="cpu"), job=job, request=image_request_from_job(job),
        project_path=tmp_path, progress=lambda *a, **k: None, cancel_requested=lambda: False,
    )
    # Applied exactly once (not once-per-pose), before the loop, under the dedicated
    # "pose" cache slot so it never collides with the text/img2img pipe bookkeeping.
    assert len(apply_calls) == 1
    assert apply_calls[0]["loras"] == loras
    assert apply_calls[0]["lora_key"] == "pose"

def test_kolors_pose_path_accepts_lokr_via_injection(monkeypatch, tmp_path):
    """sc-2252: the Kolors strict pose tier must APPLY LoKr (not just plain LoRA) on its
    torch pose pipe. _apply_loras(lora_key="pose") routes lokr_* through inject_lokr_adapter
    into the pose pipe's UNet — never load_lora_weights — and tracks the merge under the
    dedicated "pose" cache slot, distinct from the text/img2img bookkeeping."""
    lokr_file = tmp_path / "char.safetensors"
    lokr_file.write_bytes(b"x")
    monkeypatch.setattr("scene_worker.lora_adapters.adapter_network_type", lambda path: "lokr")
    injected: list[str] = []
    monkeypatch.setattr(
        "scene_worker.lora_adapters.inject_lokr_adapter",
        lambda pipe, spec, *, adapter_id: injected.append(spec.adapter_name),
    )
    pipe = FakeLokrPipe()
    adapter = KolorsDiffusersAdapter()
    request = SimpleNamespace(
        mode="character_image",
        model="kolors",
        loras=[{"id": "char", "installedPath": str(lokr_file), "weight": 0.7, "families": ["kolors"]}],
    )

    adapter._apply_loras(pipe, request, lora_key="pose")

    # LoKr injects into the pose pipe's UNet denoiser; it never loads via load_lora_weights.
    assert pipe.loaded == []
    assert injected, "LoKr must inject into the pose pipe denoiser"
    # Tracked under the dedicated pose slot — not the text/img2img slots.
    pose_state = adapter._loaded_lora_states["pose"]
    assert injected == list(pose_state.adapter_names)
    assert "text" not in adapter._loaded_lora_states
    assert "img2img" not in adapter._loaded_lora_states

def test_sdxl_adapter_applies_sdxl_lora(tmp_path):
    # A trained sdxl-family LoRA loads onto the SDXL pipeline via
    # StableDiffusionXLPipeline.load_lora_weights and its weight is applied (sc-1943).
    lora = tmp_path / "aurora_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = SdxlDiffusersAdapter()
    request = SimpleNamespace(
        model="sdxl",
        mode="text_to_image",
        loras=[
            {
                "id": "aurora_style",
                "installedPath": str(lora),
                "weight": 0.75,
                "compatibility": {"families": ["sdxl"]},
            }
        ],
    )
    adapter._apply_loras(pipe, request)
    state = adapter._loaded_lora_states["text"]
    assert [path for path, _name in pipe.loaded] == [str(lora)]
    assert pipe.set_calls == [(list(state.adapter_names), [0.75])]

def test_sdxl_adapter_rejects_incompatible_lora_family(tmp_path):
    # SDXL accepts only sdxl-family LoRAs (no extra-compatible families, unlike
    # chroma↔flux): a flux LoRA is filtered out before it can load (sc-1927/sc-1943).
    lora = tmp_path / "flux_style.safetensors"
    lora.write_bytes(b"lora")
    pipe = FakeLoraPipe()
    adapter = SdxlDiffusersAdapter()
    request = SimpleNamespace(
        model="sdxl",
        mode="text_to_image",
        loras=[
            {
                "id": "flux_style",
                "installedPath": str(lora),
                "compatibility": {"families": ["flux"]},
            }
        ],
    )
    try:
        adapter._apply_loras(pipe, request)
    except RuntimeError as exc:
        assert "not compatible with model family sdxl" in str(exc)
    else:
        raise AssertionError("SDXL must reject a LoRA whose family is not sdxl.")

def test_sdxl_backend_save_lora_writes_unet_diffusers_format(tmp_path, monkeypatch):
    # The trainer saves an sdxl-family LoRA via StableDiffusionXLPipeline's own
    # save_lora_weights(unet_lora_layers=...) — the diffusers format the inference
    # loader (load_lora_weights) round-trips. Torch-free: fake peft + pipeline.
    import sys
    import types as types_module

    fake_state = {"unet.down_blocks.0.attentions.0.lora_A.weight": "tensor"}
    fake_peft_utils = types_module.ModuleType("peft.utils")
    fake_peft_utils.get_peft_model_state_dict = lambda _module: fake_state
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)

    saved: dict[str, object] = {}

    class FakeSdxlPipeline:
        @staticmethod
        def save_lora_weights(
            output_dir, *, unet_lora_layers=None, weight_name=None, safe_serialization=None, **_kwargs
        ):
            saved["output_dir"] = output_dir
            saved["unet_lora_layers"] = unet_lora_layers
            saved["weight_name"] = weight_name
            saved["safe_serialization"] = safe_serialization

    backend = _SdxlLoraBackend()
    backend._unet = object()
    backend._pipeline = FakeSdxlPipeline()
    output_path = backend._save_lora(output_dir=str(tmp_path), file_name="aurora_style.safetensors")

    # unet_lora_layers carries the PEFT state dict; this is the only LoRA layer
    # kwarg the trainer passes (UNet-only), and the key the SDXL loader consumes.
    assert saved["unet_lora_layers"] is fake_state
    assert saved["weight_name"] == "aurora_style.safetensors"
    assert saved["safe_serialization"] is True
    assert output_path == str(tmp_path / "aurora_style.safetensors")

def test_wan_backend_save_lora_writes_transformer_diffusers_format(tmp_path, monkeypatch):
    # Default (lora) network: the Wan trainer saves via WanPipeline.save_lora_weights
    # (transformer_lora_layers=...), the diffusers format the video loader round-trips.
    # Torch-free: fake peft + pipeline (sc-2211).
    import sys
    import types as types_module

    fake_state = {"transformer.blocks.0.attn1.to_q.lora_A.weight": "tensor"}
    fake_peft_utils = types_module.ModuleType("peft.utils")
    fake_peft_utils.get_peft_model_state_dict = lambda _module: fake_state
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)

    saved: dict[str, object] = {}

    class FakeWanPipeline:
        @staticmethod
        def save_lora_weights(
            output_dir, *, transformer_lora_layers=None, weight_name=None, safe_serialization=None, **_kwargs
        ):
            saved["transformer_lora_layers"] = transformer_lora_layers
            saved["weight_name"] = weight_name
            saved["safe_serialization"] = safe_serialization

    backend = _WanLoraBackend()
    backend._transformer = object()
    backend._pipeline = FakeWanPipeline()
    output_path = backend._save_lora(output_dir=str(tmp_path), file_name="motion.safetensors")

    assert saved["transformer_lora_layers"] is fake_state
    assert saved["weight_name"] == "motion.safetensors"
    assert saved["safe_serialization"] is True
    assert output_path == str(tmp_path / "motion.safetensors")

def test_wan_backend_save_lora_lokr_routes_to_write_lokr_adapter(tmp_path, monkeypatch):
    # LoKr network (sc-2211): LoKr keys (lokr_w1/lokr_w2) aren't save_lora_weights-
    # compatible, so the Wan trainer serializes raw via write_lokr_adapter with the
    # routing metadata the video inference loader (PEFT injection) needs — exactly
    # like the SDXL/Z-Image backends. save_lora_weights must NOT be called.
    import sys
    import types as types_module

    fake_state = {"transformer.blocks.0.attn1.to_q.lokr_w1": "tensor"}
    fake_peft_utils = types_module.ModuleType("peft.utils")
    fake_peft_utils.get_peft_model_state_dict = lambda _module: fake_state
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)

    captured: dict[str, object] = {}

    def fake_write_lokr_adapter(state_dict, output_dir, file_name, **kwargs):
        captured["state_dict"] = state_dict
        captured["file_name"] = file_name
        captured["kwargs"] = kwargs
        return str(Path(output_dir) / file_name)

    monkeypatch.setattr(
        "scene_worker.training_adapters.write_lokr_adapter", fake_write_lokr_adapter
    )

    class FakeWanPipeline:
        @staticmethod
        def save_lora_weights(*_args, **_kwargs):
            raise AssertionError("LoKr must not save via save_lora_weights")

    backend = _WanLoraBackend()
    backend._transformer = object()
    backend._pipeline = FakeWanPipeline()
    backend._network_type = "lokr"
    save_kwargs = {
        "rank": 16,
        "alpha": 16,
        "decompose_factor": -1,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    }
    backend._lokr_save_kwargs = save_kwargs

    output_path = backend._save_lora(output_dir=str(tmp_path), file_name="motion.safetensors")

    assert captured["state_dict"] is fake_state
    assert captured["file_name"] == "motion.safetensors"
    assert captured["kwargs"] == save_kwargs
    assert output_path == str(tmp_path / "motion.safetensors")
