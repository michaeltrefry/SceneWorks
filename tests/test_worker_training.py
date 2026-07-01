from __future__ import annotations

from worker_runtime_shared import *

def test_gpu_worker_advertises_lora_train_without_inference_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})

    assert "lora_train" in capabilities

def test_gpu_worker_advertises_lora_train_execute_only_with_inference_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    with_backend = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "lora_train" in with_backend
    assert "lora_train_execute" in with_backend

    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: False)
    without_backend = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    # Dry-run validation stays claimable; real execution is not advertised.
    assert "lora_train" in without_backend
    assert "lora_train_execute" not in without_backend

def test_lens_train_runner_reads_network_type_and_factor():
    from scene_worker import lens_train_runner as lt

    assert lt._network_type({}) == "lora"
    assert lt._decompose_factor({}) == -1
    assert lt._network_type({"advanced": {"networkType": "LoKr"}}) == "lokr"
    assert lt._decompose_factor({"advanced": {"decomposeFactor": 8}}) == 8
    # Bad factor falls back to auto (-1).
    assert lt._decompose_factor({"advanced": {"decomposeFactor": "x"}}) == -1
    # Default targets point at the Linear `to_out.0`, not the `to_out` ModuleList
    # (PEFT errors on the ModuleList — sc-2218).
    assert lt._target_modules({}) == ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"]
    assert "to_out" not in lt._target_modules({})  # the bare ModuleList name is gone

def test_lens_train_runner_build_network_config():
    from scene_worker import lens_train_runner as lt

    fake_peft = SimpleNamespace(
        LoraConfig=lambda **kw: ("lora", kw),
        LoKrConfig=lambda **kw: ("lokr", kw),
    )
    kind, _ = lt._build_network_config(
        fake_peft, network_type="lora", rank=8, alpha=8, decompose_factor=-1,
        target_modules=["img_qkv", "txt_qkv"],
    )
    assert kind == "lora"

    kind, kwargs = lt._build_network_config(
        fake_peft, network_type="lokr", rank=8, alpha=16, decompose_factor=4,
        target_modules=["img_qkv", "txt_qkv", "to_out.0", "to_add_out"],
    )
    assert kind == "lokr"
    assert kwargs["r"] == 8 and kwargs["alpha"] == 16
    assert kwargs["decompose_factor"] == 4
    assert kwargs["target_modules"] == ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"]

def test_lora_train_dry_run_completes_with_plan_summary(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    image = tmp_path / "images" / "001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    api = _DryRunApi()
    plan = {
        "planVersion": 1,
        "dataset": {
            "datasetId": "ds_1",
            "datasetVersion": 2,
            "items": [{"imagePath": str(image), "caption": "auroraStyle portrait"}],
        },
        "target": {
            "targetId": "z_image_turbo_lora",
            "kernel": "z_image_lora",
            "baseModelPath": str(tmp_path / "uninstalled-model"),
        },
        "output": {
            "loraId": "lora_1",
            "outputDir": str(tmp_path / "loras" / "lora_1"),
            "fileName": "aurora.safetensors",
        },
    }

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="cpu"), _lora_train_job(plan))

    terminal = api.progress[-1]
    assert terminal["status"] == "completed"
    assert terminal["stage"] == "completed"
    result = terminal["result"]
    assert result["mode"] == "dry_run"
    assert result["validated"] is True
    assert result["datasetItemCount"] == 1
    assert result["loraId"] == "lora_1"
    assert result["fileName"] == "aurora.safetensors"
    # The base model is not installed yet; the dry run records that without failing.
    assert result["baseModelInstalled"] is False

def test_lora_train_dry_run_fails_cleanly_on_missing_dataset_image(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    api = _DryRunApi()
    plan = {
        "planVersion": 1,
        "dataset": {"items": [{"imagePath": str(tmp_path / "missing.png"), "caption": "x"}]},
        "target": {},
        "output": {},
    }

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="cpu"), _lora_train_job(plan))

    terminal = api.progress[-1]
    assert terminal["status"] == "failed"
    assert "missing" in terminal["error"].lower()

def test_lora_train_dry_run_fails_on_unsupported_plan_version(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    api = _DryRunApi()
    plan = {"planVersion": 999, "dataset": {"items": []}, "target": {}, "output": {}}

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="cpu"), _lora_train_job(plan))

    terminal = api.progress[-1]
    assert terminal["status"] == "failed"
    assert "version" in terminal["error"].lower()

def test_dry_run_summary_records_base_model_repo_and_install_state(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    plan = {
        "planVersion": 1,
        "dataset": {"datasetId": "ds_1", "datasetVersion": 4, "items": [{"imagePath": "x"}]},
        "target": {
            "targetId": "z_image_turbo_lora",
            "kernel": "z_image_lora",
            "baseModel": "z_image_turbo",
            "baseModelRepo": "Tongyi-MAI/Z-Image-Turbo",
            "baseModelPath": str(model_dir),
        },
        "output": {"loraId": "lora_1", "outputDir": str(tmp_path / "out"), "fileName": "a.safetensors"},
    }

    summary = dry_run_training_summary(plan, dry_run=True)

    assert summary["baseModelRepo"] == "Tongyi-MAI/Z-Image-Turbo"
    assert summary["baseModelInstalled"] is True
    assert summary["datasetVersion"] == 4

def test_validate_training_plan_rejects_bad_version_and_empty_dataset():
    with pytest.raises(ValueError, match="version"):
        validate_training_plan({"planVersion": 99, "dataset": {"items": [{"imagePath": "x"}]}})
    with pytest.raises(ValueError, match="no items"):
        validate_training_plan({"planVersion": SUPPORTED_TRAINING_PLAN_VERSION, "dataset": {"items": []}})

def test_bucket_resolution_floors_to_multiple_of_32():
    assert bucket_resolution(1024) == 1024
    assert bucket_resolution(1000) == 992
    assert bucket_resolution(20) == 32

def test_flow_matching_velocity_target_uses_negated_pipeline_sign():
    # The raw transformer output target is latents - noise, NOT noise - latents:
    # diffusers' ZImagePipeline negates the transformer output before the scheduler,
    # so the trained raw output is the negated flow velocity. Pin the sign so a
    # refactor can't silently flip the training direction.
    assert flow_matching_velocity_target(0.0, 1.0) == -1.0
    assert flow_matching_velocity_target(2.0, -1.0) == 3.0
    latents, noise = 0.7, 0.2
    assert flow_matching_velocity_target(latents, noise) == latents - noise
    assert flow_matching_velocity_target(latents, noise) == -(noise - latents)

def test_sample_training_timestep_accepts_ai_toolkit_shape_and_bias():
    torch = pytest.importorskip("torch")
    generator = torch.Generator("cpu").manual_seed(7)

    timestep = sample_training_timestep(
        torch,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
        timestep_type="sigmoid",
        timestep_bias="high_noise",
    )

    assert timestep.shape == (1,)
    assert float(timestep.item()) > 0.001
    assert float(timestep.item()) < 0.999

def test_seeded_sample_draws_on_generator_device_then_moves():
    """Regression for the MPS crash: a cpu ``torch.Generator`` cannot drive a
    ``torch.randn(..., device='mps')`` call. ``seeded_sample`` must draw on the
    generator's own device and move to the target when they differ, and pass the
    device straight through when they match. (``meta`` stands in for a non-cpu
    target so the routing is exercised without an actual GPU/MPS backend.)"""
    torch = pytest.importorskip("torch")
    generator = torch.Generator("cpu").manual_seed(11)

    seen_devices: list[str] = []

    def fake_fn(shape, *, generator, device, dtype):  # noqa: ARG001
        seen_devices.append(str(device))
        return torch.zeros(shape, dtype=dtype)

    # Matching device type → pass through, no move.
    out = seeded_sample(torch, fake_fn, (2,), generator=generator, device="cpu", dtype=torch.float32)
    assert seen_devices == ["cpu"]
    assert out.device.type == "cpu"

    # Mismatched target (cpu generator → non-cpu device) → draw on cpu, then move.
    seen_devices.clear()
    moved = seeded_sample(torch, fake_fn, (2,), generator=generator, device="meta", dtype=torch.float32)
    assert seen_devices == ["cpu"], "must generate on the generator's device, not the mismatched target"
    assert moved.device.type == "meta", "result must be moved to the requested device"

def test_build_optimizer_uses_prodigy_with_aitoolkit_lr_floor(monkeypatch):
    calls = {}

    class FakeProdigy:
        def __init__(self, params, **kwargs):
            calls["params"] = params
            calls["kwargs"] = kwargs

    def fake_import_module(name):
        if name == "torch":
            return SimpleNamespace(optim=SimpleNamespace())
        if name == "prodigyopt":
            return SimpleNamespace(Prodigy=FakeProdigy)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", fake_import_module)
    params = [object()]

    optimizer = build_optimizer("prodigyopt", params, 0.0001, 0.0001)

    assert isinstance(optimizer, FakeProdigy)
    assert calls == {"params": params, "kwargs": {"lr": 1.0, "eps": 1e-6, "weight_decay": 0.0001}}

def test_build_optimizer_uses_rose(monkeypatch):
    calls = {}

    class FakeRose:
        def __init__(self, params, **kwargs):
            calls["params"] = params
            calls["kwargs"] = kwargs

    def fake_import_module(name):
        if name == "torch":
            return SimpleNamespace(optim=SimpleNamespace())
        if name == "rose_opt":
            return SimpleNamespace(Rose=FakeRose)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", fake_import_module)
    params = [object()]

    optimizer = build_optimizer("rose", params, 0.0005, 0.01)

    assert isinstance(optimizer, FakeRose)
    # compute_dtype="fp32" is pinned to avoid Rose's fp64 default, which has no
    # MPS kernel on Apple Silicon.
    assert calls == {
        "params": params,
        "kwargs": {"lr": 0.0005, "weight_decay": 0.01, "compute_dtype": "fp32"},
    }

def test_read_run_config_defaults_network_type_to_lora():
    config = _run_config_for_network()
    assert config.network_type == "lora"
    assert config.decompose_factor == -1

def test_read_run_config_parses_lokr_network_and_factor():
    config = _run_config_for_network({"networkType": "LoKr", "decomposeFactor": 8})
    # networkType is normalized to lowercase so the trainer's equality check is stable.
    assert config.network_type == "lokr"
    assert config.decompose_factor == 8

def test_build_peft_network_config_defaults_to_lora():
    fake_peft = SimpleNamespace(
        LoraConfig=lambda **kw: ("lora", kw),
        LoKrConfig=lambda **kw: ("lokr", kw),
    )
    config = _run_config_for_network({"loraTargetModules": ["to_q", "to_v"]})
    kind, kwargs = build_peft_network_config(fake_peft, config)
    assert kind == "lora"
    assert kwargs == {
        "r": config.rank,
        "lora_alpha": config.alpha,
        "init_lora_weights": "gaussian",
        "target_modules": ["to_q", "to_v"],
    }

def test_build_peft_network_config_builds_lokr_with_decompose_factor():
    fake_peft = SimpleNamespace(
        LoraConfig=lambda **kw: ("lora", kw),
        LoKrConfig=lambda **kw: ("lokr", kw),
    )
    config = _run_config_for_network(
        {"networkType": "lokr", "decomposeFactor": 16, "loraTargetModules": ["to_q"]}
    )
    kind, kwargs = build_peft_network_config(fake_peft, config)
    assert kind == "lokr"
    assert kwargs == {
        "r": config.rank,
        "alpha": config.alpha,
        "decompose_factor": 16,
        "init_weights": True,
        "target_modules": ["to_q"],
    }

def test_write_lokr_adapter_stamps_metadata_and_serializes_cpu_tensors(monkeypatch, tmp_path):
    captured = {}

    def fake_save_file(tensors, path, metadata=None):
        captured["tensors"] = tensors
        captured["path"] = path
        captured["metadata"] = metadata
        Path(path).write_bytes(b"")

    fake_module = SimpleNamespace(save_file=fake_save_file)
    # Inject the parent package too so the function's `from safetensors.torch
    # import save_file` never touches the filesystem regardless of install state.
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(torch=fake_module))
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_module)

    class FakeTensor:
        def __init__(self):
            self.moved = []

        def detach(self):
            self.moved.append("detach")
            return self

        def cpu(self):
            self.moved.append("cpu")
            return self

        def contiguous(self):
            self.moved.append("contiguous")
            return self

    state = {"blk.lokr_w1": FakeTensor(), "blk.lokr_w2": FakeTensor()}
    path = write_lokr_adapter(
        state,
        str(tmp_path),
        "adapter.safetensors",
        rank=8,
        alpha=16,
        decompose_factor=8,
        target_modules=["to_q", "to_v"],
    )

    assert path == str(tmp_path / "adapter.safetensors")
    # Routing + reconstruction metadata: the inference loader (epic 2193) reads
    # networkType to route and rank/alpha/decomposeFactor/targetModules to rebuild
    # the matching LoKrConfig for injection.
    assert captured["metadata"] == {
        "format": "pt",
        "networkType": "lokr",
        "rank": "8",
        "alpha": "16",
        "decomposeFactor": "8",
        "targetModules": '["to_q", "to_v"]',
    }
    assert set(captured["tensors"]) == {"blk.lokr_w1", "blk.lokr_w2"}
    assert captured["tensors"]["blk.lokr_w1"].moved == ["detach", "cpu", "contiguous"]

def test_lr_schedule_updates_converts_microsteps_to_optimizer_updates():
    # accum=1 -> one optimizer update per micro-step; warmup passes through.
    assert lr_schedule_updates(1000, 1, 0) == (1000, 0)
    assert lr_schedule_updates(1000, 1, 100) == (1000, 100)
    # Gradient accumulation makes optimizer updates less frequent (ceil division),
    # and the warmup step count is converted the same way.
    assert lr_schedule_updates(1000, 4, 40) == (250, 10)
    assert lr_schedule_updates(10, 4, 0) == (3, 0)  # ceil(10 / 4)
    # Warmup is clamped strictly below the total so the body always decays.
    assert lr_schedule_updates(10, 1, 50) == (10, 9)

def test_normalize_lr_scheduler_accepts_supported_and_rejects_unknown():
    assert normalize_lr_scheduler(None) == "constant"
    assert normalize_lr_scheduler(" Cosine ") == "cosine"
    assert normalize_lr_scheduler("LINEAR") == "linear"
    for name in SUPPORTED_LR_SCHEDULERS:
        assert normalize_lr_scheduler(name) == name
    with pytest.raises(TrainingKernelError) as excinfo:
        normalize_lr_scheduler("warmup_cosine")
    assert "Unsupported lrScheduler" in str(excinfo.value)

def test_lr_decay_multiplier_curves():
    # Constant holds the base LR for the whole run.
    assert lr_decay_multiplier("constant", 0, 10, 0) == 1.0
    assert lr_decay_multiplier("constant", 5, 10, 0) == 1.0
    # Linear decays 1 -> 0 across the run.
    assert lr_decay_multiplier("linear", 0, 10, 0) == pytest.approx(1.0)
    assert lr_decay_multiplier("linear", 5, 10, 0) == pytest.approx(0.5)
    assert lr_decay_multiplier("linear", 10, 10, 0) == pytest.approx(0.0)
    # Cosine: 1 at the start, 0.5 at the midpoint, 0 at the end.
    assert lr_decay_multiplier("cosine", 0, 10, 0) == pytest.approx(1.0)
    assert lr_decay_multiplier("cosine", 5, 10, 0) == pytest.approx(0.5)
    assert lr_decay_multiplier("cosine", 10, 10, 0) == pytest.approx(0.0)
    # A linear warmup ramps to 1.0 without a dead zero-LR first step, then the
    # body schedule runs from its start (progress 0 -> multiplier 1.0).
    assert lr_decay_multiplier("cosine", 0, 10, 4) == pytest.approx(1 / 5)
    assert lr_decay_multiplier("cosine", 3, 10, 4) == pytest.approx(4 / 5)
    assert lr_decay_multiplier("cosine", 4, 10, 4) == pytest.approx(1.0)
    # Cosine and linear are different curves off the midpoint.
    assert lr_decay_multiplier("cosine", 3, 10, 0) != pytest.approx(
        lr_decay_multiplier("linear", 3, 10, 0)
    )

def test_build_lr_scheduler_rejects_unknown_before_touching_torch():
    # The name is validated before the optimizer/torch is used, so a dummy torch
    # is never dereferenced for an unknown scheduler.
    with pytest.raises(TrainingKernelError):
        build_lr_scheduler(
            SimpleNamespace(), SimpleNamespace(), "exotic", total_updates=5, warmup_updates=0
        )

def test_build_lr_scheduler_constant_is_fixed_and_cosine_linear_decay():
    torch = pytest.importorskip("torch")

    def make_optimizer():
        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([param], lr=0.1)

    # Plain constant (no warmup) returns no scheduler: the LR stays exactly fixed,
    # matching every pre-scheduler training run.
    optimizer = make_optimizer()
    assert (
        build_lr_scheduler(torch, optimizer, "constant", total_updates=10, warmup_updates=0)
        is None
    )

    def run(name):
        optimizer = make_optimizer()
        scheduler = build_lr_scheduler(torch, optimizer, name, total_updates=10, warmup_updates=0)
        assert scheduler is not None
        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        return lrs

    cosine = run("cosine")
    linear = run("linear")

    # Both start at the base LR, decay monotonically, and reach ~0 at the end.
    for lrs in (cosine, linear):
        assert lrs[0] == pytest.approx(0.1)
        assert all(later <= earlier + 1e-12 for earlier, later in zip(lrs, lrs[1:]))
        assert lrs[-1] == pytest.approx(0.0, abs=1e-7)
        assert lrs[5] < lrs[0]
    # Cosine and linear coincide at the exact midpoint (both 0.5) but differ
    # off-center, so compare away from it.
    assert cosine[3] != pytest.approx(linear[3])

def test_build_lr_scheduler_applies_linear_warmup_then_holds_for_constant():
    torch = pytest.importorskip("torch")
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)

    # Constant + warmup still builds a scheduler that ramps the LR in, then holds.
    scheduler = build_lr_scheduler(
        torch, optimizer, "constant", total_updates=10, warmup_updates=4
    )
    assert scheduler is not None

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Linear ramp 0.02 -> 0.04 -> 0.06 -> 0.08 -> 0.10, then held at the base LR.
    assert lrs[0] == pytest.approx(0.02)
    assert lrs[1] > lrs[0]
    assert lrs[4] == pytest.approx(0.1)
    assert lrs[5] == pytest.approx(0.1)

def test_read_run_config_parses_lr_scheduler_and_warmup():
    config = read_run_config(
        {"config": {"steps": 1200, "advanced": {"lrScheduler": "cosine", "lrWarmupSteps": 100}}}
    )
    assert config.lr_scheduler == "cosine"
    assert config.lr_warmup_steps == 100

def test_read_run_config_defaults_lr_scheduler_to_constant():
    config = read_run_config({"config": {}})
    assert config.lr_scheduler == "constant"
    assert config.lr_warmup_steps == 0

def test_read_run_config_defaults_weight_noise_sigma_to_zero():
    config = read_run_config({"config": {}})
    assert config.weight_noise_sigma == 0.0

def test_read_run_config_parses_weight_noise_sigma():
    config = read_run_config(
        {"config": {"advanced": {"weightNoiseSigma": 0.00125}}}
    )
    assert config.weight_noise_sigma == pytest.approx(0.00125)

def test_read_run_config_clamps_negative_weight_noise_sigma_to_zero():
    config = read_run_config(
        {"config": {"advanced": {"weightNoiseSigma": -0.01}}}
    )
    assert config.weight_noise_sigma == 0.0

def test_apply_weight_noise_is_no_op_when_sigma_is_zero():
    torch = pytest.importorskip("torch")

    param = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.SGD([param], lr=0.0)
    apply_weight_noise(torch, optimizer, 0.0)
    assert torch.equal(param.detach(), torch.zeros(4))

def test_apply_weight_noise_perturbs_with_expected_magnitude():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.zeros(4096))
    optimizer = torch.optim.SGD([param], lr=0.0)
    sigma = 0.01
    apply_weight_noise(torch, optimizer, sigma)
    # Population std of N(0, sigma) ~= sigma; allow a generous bound for 4096 samples.
    std = float(param.detach().std())
    assert std == pytest.approx(sigma, rel=0.1)
    # Mean should be near zero — perturbation is centered.
    assert abs(float(param.detach().mean())) < sigma

def test_apply_weight_noise_skips_frozen_params():
    torch = pytest.importorskip("torch")

    trainable = torch.nn.Parameter(torch.zeros(8))
    frozen = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
    optimizer = torch.optim.SGD([{"params": [trainable, frozen]}], lr=0.0)
    apply_weight_noise(torch, optimizer, 0.01)
    assert torch.equal(frozen.detach(), torch.zeros(8))
    assert not torch.equal(trainable.detach(), torch.zeros(8))

def test_z_image_lora_backend_activates_default_adapter():
    class FakeTransformer:
        def __init__(self):
            self.adapter_name = None

        def set_adapter(self, name):
            self.adapter_name = name

    transformer = FakeTransformer()

    _ZImageLoraBackend()._activate_lora_adapter(transformer)

    assert transformer.adapter_name == "default"

def test_read_run_config_parses_training_adapter():
    config = read_run_config(
        {
            "config": {
                "advanced": {
                    "trainingAdapterRepo": "ostris/zimage_turbo_training_adapter",
                    "trainingAdapterVersion": "v2-default",
                }
            }
        }
    )

    assert config.training_adapter_repo == "ostris/zimage_turbo_training_adapter"
    assert config.training_adapter_version == "v2-default"

def test_read_run_config_training_adapter_absent_is_none():
    config = read_run_config({"config": {"advanced": {}}})

    assert config.training_adapter_repo is None
    assert config.training_adapter_version is None

def test_training_adapter_weight_name_maps_versions():
    assert training_adapter_weight_name("v1") == "zimage_turbo_training_adapter_v1.safetensors"
    assert training_adapter_weight_name("v1-default") == "zimage_turbo_training_adapter_v1.safetensors"
    assert training_adapter_weight_name("v2-default") == "zimage_turbo_training_adapter_v2.safetensors"
    # Unknown / empty defaults to v2 (the SceneWorks preset default).
    assert training_adapter_weight_name(None) == "zimage_turbo_training_adapter_v2.safetensors"
    assert training_adapter_weight_name("") == "zimage_turbo_training_adapter_v2.safetensors"

def test_resolve_training_adapter_source_prefers_cached_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    repo = "ostris/zimage_turbo_training_adapter"
    repo_root = tmp_path / "hub" / "models--ostris--zimage_turbo_training_adapter"
    snapshot = repo_root / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir(parents=True)
    (repo_root / "refs" / "main").write_text("deadbeef", encoding="utf-8")
    weight_file = snapshot / "zimage_turbo_training_adapter_v2.safetensors"
    weight_file.write_bytes(b"weights")

    load_target, weight_name = resolve_training_adapter_source(repo, "v2-default")

    assert load_target == str(weight_file)
    assert weight_name == "zimage_turbo_training_adapter_v2.safetensors"

def test_resolve_training_adapter_source_falls_back_to_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    repo = "ostris/zimage_turbo_training_adapter"

    load_target, weight_name = resolve_training_adapter_source(repo, "v1")

    assert load_target == repo
    assert weight_name == "zimage_turbo_training_adapter_v1.safetensors"

def test_apply_training_adapter_fuses_and_unloads(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent-cache")
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    calls = []

    class FakePipe:
        def load_lora_weights(self, target, *, weight_name=None, adapter_name=None):
            calls.append(("load", target, weight_name, adapter_name))

        def fuse_lora(self):
            calls.append(("fuse",))

        def unload_lora_weights(self):
            calls.append(("unload",))

    config = read_run_config(
        {
            "config": {
                "advanced": {
                    "trainingAdapterRepo": "ostris/zimage_turbo_training_adapter",
                    "trainingAdapterVersion": "v2-default",
                }
            }
        }
    )

    weight_name = _ZImageLoraBackend()._apply_training_adapter(
        FakePipe(), config, lambda *args, **kwargs: None
    )

    assert weight_name == "zimage_turbo_training_adapter_v2.safetensors"
    assert [call[0] for call in calls] == ["load", "fuse", "unload"]
    load_call = calls[0]
    assert load_call[1] == "ostris/zimage_turbo_training_adapter"
    assert load_call[2] == "zimage_turbo_training_adapter_v2.safetensors"
    assert load_call[3] == "dedistill"

def test_apply_training_adapter_noop_without_repo():
    calls = []

    class FakePipe:
        def load_lora_weights(self, *args, **kwargs):
            calls.append("load")

        def fuse_lora(self):
            calls.append("fuse")

    config = read_run_config({"config": {"advanced": {}}})

    result = _ZImageLoraBackend()._apply_training_adapter(
        FakePipe(), config, lambda *args, **kwargs: None
    )

    assert result is None
    assert calls == []

def test_apply_training_adapter_raises_on_failure(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent-cache")
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    class FakePipe:
        def load_lora_weights(self, *args, **kwargs):
            raise RuntimeError("no such adapter")

        def fuse_lora(self):
            pass

    config = read_run_config(
        {
            "config": {
                "advanced": {
                    "trainingAdapterRepo": "ostris/zimage_turbo_training_adapter",
                }
            }
        }
    )

    with pytest.raises(TrainingKernelError, match="de-distill"):
        _ZImageLoraBackend()._apply_training_adapter(
            FakePipe(), config, lambda *args, **kwargs: None
        )

def test_read_run_config_defaults_lora_target_modules_and_parses_advanced():
    config = read_run_config(
        {
            "config": {
                "rank": 8,
                "alpha": 12,
                "learningRate": 0.0003,
                "steps": 500,
                "saveEvery": 100,
                "optimizer": "adamw8bit",
                "advanced": {
                    "mixedPrecision": "bf16",
                    "weightDecay": 0.0001,
                    "timestepType": "sigmoid",
                    "timestepBias": "high_noise",
                    "lossType": "mse",
                    "gradientCheckpointing": True,
                },
            }
        }
    )

    assert config.rank == 8
    assert config.alpha == 12
    assert config.steps == 500
    assert config.save_every == 100
    assert config.mixed_precision == "bf16"
    assert config.weight_decay == 0.0001
    assert config.timestep_type == "sigmoid"
    assert config.timestep_bias == "high_noise"
    assert config.loss_type == "mse"
    assert config.gradient_checkpointing is True
    assert config.lora_target_modules == ["to_q", "to_k", "to_v", "to_out.0"]
    assert config.sample_steps == 9
    assert config.sample_guidance_scale == 0.0

def test_read_run_config_parses_sample_render_settings():
    config = read_run_config(
        {
            "config": {
                "advanced": {
                    "sampleEvery": 50,
                    "sampleSteps": 12,
                    "sampleGuidanceScale": 1.25,
                    "samplePrompts": ["miraStyle portrait"],
                }
            }
        }
    )

    assert config.sample_every == 50
    assert config.sample_steps == 12
    assert config.sample_guidance_scale == 1.25
    assert config.sample_prompts == ["miraStyle portrait"]
    # sc-8671: absent sampleCount defaults to 4 (the historical fixed cap).
    assert config.sample_count == 4

def test_read_run_config_caps_sample_prompts_at_sample_count():
    # sc-8671: sampleCount caps how many previews render (one per prompt, truncated,
    # never padded). A pool larger than the count is truncated...
    capped = read_run_config(
        {
            "config": {
                "advanced": {
                    "samplePrompts": ["p1", "p2", "p3", "p4", "p5"],
                    "sampleCount": 2,
                }
            }
        }
    )
    assert capped.sample_count == 2
    assert capped.sample_prompts == ["p1", "p2"]

    # ...and a count at/above the pool size leaves every prompt intact (no duplication).
    uncapped = read_run_config(
        {
            "config": {
                "advanced": {
                    "samplePrompts": ["p1", "p2"],
                    "sampleCount": 8,
                }
            }
        }
    )
    assert uncapped.sample_prompts == ["p1", "p2"]

def test_read_run_config_defaults_z_image_samples_to_turbo_guidance():
    config = read_run_config(
        {
            "target": {"kernel": "z_image_lora", "baseModel": "z_image_turbo"},
            "config": {"advanced": {"samplePrompts": ["miraStyle portrait"]}},
        }
    )

    assert config.sample_guidance_scale == 1.0

def test_z_image_lora_backend_generates_samples_with_turbo_guidance(tmp_path):
    calls = []

    class FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeGenerator:
        def __init__(self, device):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakeTorch:
        def no_grad(self):
            return FakeNoGrad()

        def Generator(self, device):
            return FakeGenerator(device)

    class FakeImage:
        def convert(self, mode):
            assert mode == "RGB"
            return self

        def save(self, path):
            Path(path).write_bytes(b"png")

    class FakePipe:
        def __call__(
            self,
            *,
            prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            generator,
        ):
            calls.append(
                {
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": generator.seed,
                }
            )
            return SimpleNamespace(images=[FakeImage()])

    class FakeTransformer:
        training = True

        def set_adapter(self, _name):
            pass

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    backend = _ZImageLoraBackend()
    backend._torch = FakeTorch()
    backend._pipeline = FakePipe()
    backend._transformer = FakeTransformer()
    backend._device = "cpu"
    config = read_run_config(
        {
            "config": {
                "seed": 7,
                "advanced": {
                    "sampleSteps": 11,
                    "samplePrompts": ["miraStyle portrait"],
                },
            },
            "target": {"kernel": "z_image_lora", "baseModel": "z_image_turbo"},
            "output": {"triggerWords": ["miraStyle"]},
        }
    )

    samples = backend.generate_samples(
        step=4,
        prompts=config.sample_prompts,
        output_dir=str(tmp_path),
        file_name="mira.safetensors",
        plan={"dataset": {"rootPath": str(tmp_path / "training" / "datasets" / "ds_1")}},
        config=config,
    )

    assert calls[0]["num_inference_steps"] == 11
    assert calls[0]["guidance_scale"] == 1.0
    assert calls[0]["seed"] == 11
    assert samples[0]["sampleSource"] == "live_adapter"
    assert samples[0]["numInferenceSteps"] == 11
    assert samples[0]["guidanceScale"] == 1.0

def test_create_training_kernel_resolves_known_and_rejects_unknown():
    assert isinstance(create_training_kernel("z_image_lora"), ZImageLoraTrainer)
    assert isinstance(create_training_kernel("sdxl_lora"), SdxlLoraTrainer)
    assert isinstance(create_training_kernel("kolors_lora"), KolorsLoraTrainer)
    assert isinstance(create_training_kernel("wan_lora"), WanLoraTrainer)
    assert isinstance(create_training_kernel("wan_moe_lora"), WanMoeLoraTrainer)
    assert isinstance(create_training_kernel("lens_lora"), LensLoraTrainer)
    with pytest.raises(TrainingKernelError, match="No training kernel"):
        create_training_kernel("not_a_kernel")

def test_kolors_lora_trainer_reuses_sdxl_backend_with_kolors_seams():
    # KolorsLoraTrainer (epic 1929) is a thin extension of the generic SDXL-UNet
    # trainer: same orchestration + SDXL training loop, swapping only the pipeline
    # class + ChatGLM3 prompt encoder. LoKr is inherited from the SDXL backend.
    trainer = create_training_kernel("kolors_lora")
    assert isinstance(trainer, SdxlLoraTrainer)
    assert trainer.kernel_id == "kolors_lora"
    backend = trainer._create_backend()
    assert isinstance(backend, _KolorsLoraBackend)
    assert isinstance(backend, _SdxlLoraBackend)  # inherits the LoKr-wired save/load
    assert backend.kernel_id == "kolors_lora"
    assert backend.pipeline_class_name == "KolorsPipeline"

def test_kolors_encode_prompt_uses_chatglm_seam():
    # The only forward-pass seam: Kolors uses a single ChatGLM3 encoder (no SDXL
    # prompt_2) at GLM sequence length 256; the 4-tuple return order matches SDXL.
    captured: dict[str, object] = {}

    def fake_encode_prompt(**kwargs):
        captured.update(kwargs)
        return ("PROMPT_EMBEDS", "NEG", "POOLED", "NEG_POOLED")

    pipe = SimpleNamespace(encode_prompt=fake_encode_prompt)
    backend = _KolorsLoraBackend()
    prompt_embeds, pooled = backend._encode_prompt(pipe, "a calico cat", "cpu")

    assert prompt_embeds == "PROMPT_EMBEDS"
    assert pooled == "POOLED"
    assert captured["prompt"] == "a calico cat"
    assert captured["max_sequence_length"] == 256
    assert captured["do_classifier_free_guidance"] is False
    # Kolors has no second CLIP encoder — the SDXL `prompt_2` arg must be absent.
    assert "prompt_2" not in captured

def test_kolors_backend_releases_chatglm_encoder_when_not_sampling(monkeypatch):
    # ChatGLM3-6B is only needed to cache prompt embeddings. With no live sampling
    # the encoder is released after caching (Mac memory envelope, epic 1929); with
    # sampling on it's retained because generate_samples re-encodes prompts.
    monkeypatch.setattr(_SdxlLoraBackend, "prepare_dataset", lambda self, **kw: {"itemCount": 0})
    monkeypatch.setattr("scene_worker.training_adapters.empty_torch_cache", lambda _torch: None)

    def run(sample_every):
        backend = _KolorsLoraBackend()
        encoder = object()
        backend._pipeline = SimpleNamespace(text_encoder=encoder, text_encoder_2=None)
        backend.prepare_dataset(
            items=[],
            config=SimpleNamespace(sample_every=sample_every),
            progress=lambda *a, **k: None,
            cancel_requested=lambda: False,
        )
        return backend._pipeline.text_encoder, encoder

    released, _ = run(0)
    assert released is None  # released when not sampling
    retained, encoder = run(250)
    assert retained is encoder  # kept for live sampling

def test_wan_lora_trainer_reuses_zimage_orchestration_with_wan_backend():
    # WanLoraTrainer subclasses ZImageLoraTrainer (shared staged orchestration)
    # and only swaps the kernel id + the torch Wan backend it builds. The 14B MoE
    # trainer (sc-1953) extends this for the two-expert case.
    trainer = create_training_kernel("wan_lora")
    assert isinstance(trainer, ZImageLoraTrainer)
    assert trainer.kernel_id == "wan_lora"
    backend = trainer._create_backend()
    assert isinstance(backend, _WanLoraBackend)
    assert backend.kernel_id == "wan_lora"
    # Implements the full TrainingBackend protocol.
    for method in (
        "load",
        "prepare_dataset",
        "train_step",
        "save_checkpoint",
        "generate_samples",
        "save_final",
        "cleanup",
        "loaded_models",
    ):
        assert callable(getattr(backend, method)), method

def test_wan_lora_backend_read_run_config_uses_wan_target_modules():
    # The Rust wan_lora target declares the Wan transformer attention modules; the
    # kernel reads them straight from the plan's advanced config.
    plan = {
        "config": {
            "rank": 32,
            "steps": 1500,
            "advanced": {"loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"]},
        }
    }
    config = read_run_config(plan)
    assert list(config.lora_target_modules) == ["to_q", "to_k", "to_v", "to_out.0"]

def test_wan_moe_lora_trainer_extends_wan_backend():
    # WanMoeLoraTrainer subclasses the dense Wan trainer's backend for the A14B
    # two-expert case; it shares the orchestration and only swaps the backend.
    trainer = create_training_kernel("wan_moe_lora")
    assert isinstance(trainer, ZImageLoraTrainer)
    assert trainer.kernel_id == "wan_moe_lora"
    backend = trainer._create_backend()
    assert isinstance(backend, _WanMoeLoraBackend)
    assert isinstance(backend, _WanLoraBackend)
    assert backend.kernel_id == "wan_moe_lora"
    for method in ("load", "prepare_dataset", "train_step", "save_final", "cleanup"):
        assert callable(getattr(backend, method)), method

def test_wan_moe_lora_backend_parses_gguf_quant_spec_and_boundary():
    backend = _WanMoeLoraBackend()
    # Default boundary (A14B = 0.875) before any load.
    assert backend._boundary == 0.875
    # A complete gguf baseQuantization advanced block parses into an expert spec.
    gguf = read_run_config(
        {
            "config": {
                "advanced": {
                    "baseQuantization": {
                        "format": "gguf",
                        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
                        "highNoiseFile": "HighNoise/hi.gguf",
                        "lowNoiseFile": "LowNoise/lo.gguf",
                    }
                }
            }
        }
    )
    spec = backend._quant_spec(gguf)
    assert spec == {
        "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF",
        "highNoiseFile": "HighNoise/hi.gguf",
        "lowNoiseFile": "LowNoise/lo.gguf",
    }
    # No quant block -> bf16 path (None); an incomplete block is ignored.
    assert backend._quant_spec(read_run_config({"config": {"advanced": {}}})) is None
    incomplete = read_run_config(
        {"config": {"advanced": {"baseQuantization": {"format": "gguf", "repo": "R"}}}}
    )
    assert backend._quant_spec(incomplete) is None

def test_sdxl_lora_trainer_reuses_zimage_orchestration_with_sdxl_backend():
    # SdxlLoraTrainer subclasses ZImageLoraTrainer (shared staged orchestration)
    # and only swaps the kernel id + the backend it builds.
    trainer = create_training_kernel("sdxl_lora")
    assert isinstance(trainer, ZImageLoraTrainer)
    assert trainer.kernel_id == "sdxl_lora"
    backend = trainer._create_backend()
    assert isinstance(backend, _SdxlLoraBackend)
    # Extension seams epic 1929 (Kolors) overrides: the pipeline class + the
    # prompt encoder. Everything else is shared.
    assert backend.kernel_id == "sdxl_lora"
    assert backend.pipeline_class_name == "StableDiffusionXLPipeline"
    assert backend.load_variant == "fp16"

def test_sdxl_lora_backend_read_run_config_uses_sdxl_target_modules():
    # The Rust sdxl_lora target declares the SDXL UNet attention modules; the
    # kernel reads them straight from the plan's advanced config.
    plan = {
        "config": {
            "rank": 16,
            "steps": 1500,
            "advanced": {"loraTargetModules": ["to_q", "to_k", "to_v", "to_out.0"]},
        }
    }
    config = read_run_config(plan)
    assert list(config.lora_target_modules) == ["to_q", "to_k", "to_v", "to_out.0"]

def test_lens_trainer_drives_sidecar_and_shapes_result(tmp_path, monkeypatch):
    import subprocess as _subprocess

    plan = _lens_train_plan(tmp_path, steps=4)
    monkeypatch.setattr(LensLoraTrainer, "_sidecar_available", lambda self: True)
    monkeypatch.setattr(_subprocess, "Popen", _FakeLensSidecarPopen)

    events = []
    result = LensLoraTrainer().train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda status, stage, value, message, result=None: events.append((status, stage)),
        cancel_requested=lambda: False,
    )

    stages = {stage for _status, stage in events}
    statuses = {status for status, _stage in events}
    # The driver maps the sidecar's JSONL events onto valid JobStatus bands; in
    # particular caching runs under "running" (not the invalid "caching").
    assert {"caching_latents", "training", "saving"}.issubset(stages)
    assert statuses <= _VALID_JOB_STATUSES
    assert result["mode"] == "train"
    assert result["kernel"] == "lens_lora"
    assert result["stepsCompleted"] == 4
    assert result["outputPath"] == os.path.join(plan["output"]["outputDir"], "aurora.safetensors")
    assert result["baseModelSource"] == "SceneWorks/Lens"
    assert result["triggerWords"] == ["auroraStyle"]
    assert os.path.exists(result["outputPath"])

def test_lens_trainer_requires_sidecar(tmp_path, monkeypatch):
    plan = _lens_train_plan(tmp_path)
    monkeypatch.setattr(LensLoraTrainer, "_sidecar_available", lambda self: False)
    with pytest.raises(TrainingKernelError, match="Lens sidecar venv"):
        LensLoraTrainer().train(
            settings=SimpleNamespace(worker_id="w", gpu_id="0"),
            plan=plan,
            progress=lambda *args, **kwargs: None,
            cancel_requested=lambda: False,
        )

def test_lens_device_hint_delegates_to_select_torch_device(monkeypatch):
    # On a real worker the driver has torch in the main venv, so the sidecar
    # device hint must come from select_torch_device — picking "mps" on Apple
    # Silicon exactly like the Lens inference adapter, not a hardcoded "cuda".
    captured = {}

    def _fake_import(name):
        captured["imported"] = name
        return SimpleNamespace(__name__="torch")

    def _fake_select(torch, gpu_id):
        captured["gpu_id"] = gpu_id
        return "mps"

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", _fake_import)
    monkeypatch.setattr("scene_worker.training_adapters.select_torch_device", _fake_select)

    device = LensLoraTrainer._device_hint(SimpleNamespace(gpu_id="0"))
    assert device == "mps"
    assert captured["imported"] == "torch"
    assert captured["gpu_id"] == "0"

def test_lens_device_hint_falls_back_to_mps_without_torch(monkeypatch):
    # If torch is somehow unimportable in the main venv, the hint still resolves
    # sensibly from the platform: Apple Silicon -> "mps", explicit cpu -> "cpu".
    def _no_torch(name):
        raise ImportError("no torch in this venv")

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", _no_torch)
    monkeypatch.setattr("scene_worker.training_adapters.sys.platform", "darwin")
    monkeypatch.setattr("scene_worker.training_adapters.platform.machine", lambda: "arm64")

    assert LensLoraTrainer._device_hint(SimpleNamespace(gpu_id="0")) == "mps"
    assert LensLoraTrainer._device_hint(SimpleNamespace(gpu_id="cpu")) == "cpu"

def test_lens_device_hint_falls_back_to_cuda_off_apple_silicon(monkeypatch):
    def _no_torch(name):
        raise ImportError("no torch in this venv")

    monkeypatch.setattr("scene_worker.training_adapters.importlib.import_module", _no_torch)
    monkeypatch.setattr("scene_worker.training_adapters.sys.platform", "linux")
    monkeypatch.setattr("scene_worker.training_adapters.platform.machine", lambda: "x86_64")

    assert LensLoraTrainer._device_hint(SimpleNamespace(gpu_id="1")) == "cuda:1"
    assert LensLoraTrainer._device_hint(SimpleNamespace(gpu_id=None)) == "cuda"

def test_lens_trainer_passes_resolved_device_to_sidecar(tmp_path, monkeypatch):
    # The device the driver resolves must reach the sidecar spec verbatim, so a
    # Mac run actually trains on "mps" instead of erroring on a "cuda" hint.
    import subprocess as _subprocess

    captured = {}

    class _RecordingPopen(_FakeLensSidecarPopen):
        def __init__(self, cmd, env=None, stdout=None, stderr=None):
            spec = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
            captured["device"] = spec["device"]
            super().__init__(cmd, env=env, stdout=stdout, stderr=stderr)

    plan = _lens_train_plan(tmp_path, steps=2)
    monkeypatch.setattr(LensLoraTrainer, "_sidecar_available", lambda self: True)
    monkeypatch.setattr(LensLoraTrainer, "_device_hint", staticmethod(lambda settings: "mps"))
    monkeypatch.setattr(_subprocess, "Popen", _RecordingPopen)

    LensLoraTrainer().train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda *args, **kwargs: None,
        cancel_requested=lambda: False,
    )
    assert captured["device"] == "mps"

def test_resolve_pretrained_source_prefers_loadable_model_dir(tmp_path):
    model_dir = tmp_path / "models" / "z_image"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    source = resolve_pretrained_source(
        {"baseModelPath": str(model_dir), "baseModelRepo": "Tongyi-MAI/Z-Image-Turbo"}
    )

    assert source == str(model_dir)

def test_resolve_pretrained_source_uses_hf_cache_snapshot(tmp_path):
    cache_root = tmp_path / "hub"
    snapshot = write_huggingface_cache_resource(
        cache_root, "Tongyi-MAI/Z-Image-Turbo", "model_index.json", refs_main=True
    )
    repo_root = snapshot.parent.parent

    source = resolve_pretrained_source({"baseModelPath": str(repo_root)})

    assert source == str(snapshot)

def test_resolve_pretrained_source_falls_back_to_repo_when_path_missing(tmp_path):
    source = resolve_pretrained_source(
        {"baseModelPath": str(tmp_path / "absent"), "baseModelRepo": "Tongyi-MAI/Z-Image-Turbo"}
    )

    assert source == "Tongyi-MAI/Z-Image-Turbo"

def test_z_image_trainer_runs_stages_checkpoints_and_saves(tmp_path):
    plan = _real_train_plan(tmp_path, steps=4, save_every=2)
    backend = FakeTrainingBackend()
    trainer = ZImageLoraTrainer(backend=backend)
    events_log = []

    result = trainer.train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda status, stage, value, message: events_log.append((status, stage)),
        cancel_requested=lambda: False,
    )

    stages = [stage for _status, stage in events_log]
    statuses = {status for status, _stage in events_log}
    assert backend.events[0] == "load"
    assert ("step", 4) in backend.events
    assert {"loading_model", "caching_latents", "training", "saving"}.issubset(set(stages))
    # Every emitted status must be a valid JobStatus; the Rust API rejects others.
    # In particular, caching runs under "running" (not the invalid "caching").
    assert statuses <= _VALID_JOB_STATUSES
    assert ("running", "caching_latents") in events_log
    assert result["mode"] == "train"
    assert result["stepsCompleted"] == 4
    assert result["outputPath"] == backend.saved == os.path.join(plan["output"]["outputDir"], "mira.safetensors")
    assert result["triggerWords"] == ["miraStyle"]
    # save_every=2, steps=4 -> a single mid-run checkpoint at step 2 (step 4 is final).
    assert backend.checkpoints == [os.path.join(plan["output"]["outputDir"], "ckpt-2.safetensors")]
    assert backend.cleaned is True

def test_z_image_trainer_emits_training_samples_on_sample_cadence(tmp_path):
    plan = _real_train_plan(tmp_path, steps=4, save_every=0, sample_every=2)
    backend = FakeTrainingBackend()
    trainer = ZImageLoraTrainer(backend=backend)
    progress_results = []

    result = trainer.train(
        settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        plan=plan,
        progress=lambda *args: progress_results.append(args[4]) if len(args) > 4 else None,
        cancel_requested=lambda: False,
    )

    assert ("sample", 2) in backend.events
    assert ("sample", 4) in backend.events
    assert len(result["latestTrainingSamples"]) == 4
    assert result["latestTrainingSamples"][0]["step"] == 4
    assert result["samplePrompts"][0].startswith("miraStyle")
    assert result["sampleSettings"] == {
        "numInferenceSteps": 9,
        "guidanceScale": 1.0,
        "sampleSource": "live_adapter",
    }
    sample_updates = [payload for payload in progress_results if payload]
    assert sample_updates[-1]["latestTrainingSamples"][0]["relativePath"].startswith("loras/lora_1/samples/")
    assert sample_updates[-1]["sampleSettings"]["guidanceScale"] == 1.0

def test_z_image_trainer_cancels_and_skips_save(tmp_path):
    plan = _real_train_plan(tmp_path, steps=10, save_every=0)
    backend = FakeTrainingBackend()
    trainer = ZImageLoraTrainer(backend=backend)
    checks = {"count": 0}

    def cancel_requested():
        checks["count"] += 1
        return checks["count"] > 3

    with pytest.raises(InterruptedError):
        trainer.train(
            settings=SimpleNamespace(worker_id="worker-1", gpu_id="0"),
            plan=plan,
            progress=lambda *args: None,
            cancel_requested=cancel_requested,
        )

    assert backend.saved is None
    assert backend.cleaned is True

def test_ltx_mlx_lora_kernel_is_retired_from_python():
    # Epic 3039 (sc-3049): native MLX LTX LoRA training moved to the Rust mlx-gen
    # engine. The Python kernel was removed, so resolving it now raises — the Rust
    # mlx worker is the sole LTX-training path (routing keeps `ltx_mlx_lora` off
    # non-mlx workers; see jobs_store::training_kernel_is_mlx_only).
    with pytest.raises(TrainingKernelError):
        create_training_kernel("ltx_mlx_lora")

def test_run_lora_train_job_executes_real_run(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None, peaks=None: callback(lambda: False),
    )
    backend = FakeTrainingBackend()
    trainer = ZImageLoraTrainer(backend=backend)
    monkeypatch.setattr("scene_worker.runtime.create_training_kernel", lambda _kernel_id: trainer)

    api = _DryRunApi()
    plan = _real_train_plan(tmp_path, steps=2, save_every=0)
    job = {"id": "job-train-real", "type": "lora_train", "payload": {"dryRun": False, "plan": plan}}

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="0"), job)

    terminal = api.progress[-1]
    assert terminal["status"] == "completed"
    assert terminal["stage"] == "completed"
    assert terminal["result"]["mode"] == "train"
    assert terminal["result"]["fileName"] == "mira.safetensors"
    assert backend.saved is not None

def test_run_lora_train_job_marks_canceled(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None, peaks=None: callback(lambda: False),
    )

    class CancelingTrainer:
        kernel_id = "z_image_lora"

        def loaded_models(self):
            return []

        def train(self, *, settings, plan, progress, cancel_requested):
            raise InterruptedError("LoRA training canceled by user.")

    monkeypatch.setattr("scene_worker.runtime.create_training_kernel", lambda _kernel_id: CancelingTrainer())

    api = _DryRunApi()
    job = {
        "id": "job-train-cancel",
        "type": "lora_train",
        "payload": {"dryRun": False, "plan": {"planVersion": 1, "target": {"kernel": "z_image_lora"}}},
    }

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="0"), job)

    terminal = api.progress[-1]
    assert terminal["status"] == "canceled"
    assert "canceled" in terminal["message"].lower()

def test_run_lora_train_job_reports_friendly_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("scene_worker.runtime.emit", lambda payload: None)
    monkeypatch.setattr("scene_worker.runtime.gpu_utilization", lambda _gpu_id: None)
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda api, settings, job_id, status, callback, *, loaded_models, on_force_terminate=None, peaks=None: callback(lambda: False),
    )

    class FailingTrainer:
        kernel_id = "z_image_lora"

        def loaded_models(self):
            return []

        def train(self, *, settings, plan, progress, cancel_requested):
            raise RuntimeError("Repository not found: Tongyi-MAI/Z-Image-Turbo")

    monkeypatch.setattr("scene_worker.runtime.create_training_kernel", lambda _kernel_id: FailingTrainer())

    api = _DryRunApi()
    job = {
        "id": "job-train-fail",
        "type": "lora_train",
        "payload": {"dryRun": False, "plan": {"planVersion": 1, "target": {"kernel": "z_image_lora"}}},
    }

    run_lora_train_job(api, SimpleNamespace(worker_id="worker-1", gpu_id="0"), job)

    terminal = api.progress[-1]
    assert terminal["status"] == "failed"
    assert "model files were not available" in terminal["message"].lower()

def test_lens_trainer_discards_scratch_dir(tmp_path):
    trainer = LensLoraTrainer()
    scratch = tmp_path / "lens_train_abc"
    scratch.mkdir()
    (scratch / "spec.json").write_text("{}", encoding="utf-8")
    trainer._scratch_dir = scratch

    trainer.discard_temp_outputs()

    assert not scratch.exists()
    # Lazily-set attr cleared; a second call is a harmless no-op.
    trainer.discard_temp_outputs()

