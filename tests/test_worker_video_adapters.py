from __future__ import annotations

from worker_runtime_shared import *

def _write_video_asset_sidecar(project_path, asset_id, media_path):
    sidecar_path = project_path / "assets" / "videos" / f"{asset_id}.sceneworks.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps({"id": asset_id, "file": {"path": media_path}}),
        encoding="utf-8",
    )
    return sidecar_path

def test_video_source_asset_media_path_rejects_sidecar_paths_outside_project(tmp_path):
    from scene_worker.video_adapters import load_source_video_frames, source_asset_media_path

    in_project_media = tmp_path / "assets" / "videos" / "clip.mp4"
    in_project_media.parent.mkdir(parents=True, exist_ok=True)
    in_project_media.write_bytes(b"not-a-real-video")
    _write_video_asset_sidecar(tmp_path, "asset_inside", "assets/videos/clip.mp4")

    assert source_asset_media_path(tmp_path, "asset_inside") == in_project_media.resolve()

    outside_media = tmp_path.parent / f"{tmp_path.name}-outside.mp4"
    outside_media.write_bytes(b"outside")
    _write_video_asset_sidecar(tmp_path, "asset_absolute_escape", str(outside_media))
    _write_video_asset_sidecar(tmp_path, "asset_relative_escape", f"../{outside_media.name}")

    assert source_asset_media_path(tmp_path, "asset_absolute_escape") is None
    assert source_asset_media_path(tmp_path, "asset_relative_escape") is None
    with pytest.raises(RuntimeError, match="outside the project"):
        load_source_video_frames(tmp_path, "asset_absolute_escape", 64, 64, 1)

def test_ltx_mps_gating_leaves_cuda_path_untouched():
    gating = ltx_mps_gating(cuda_available=True, device_str="cuda")
    assert gating == {
        "device": None,
        "disable_fp8": False,
        "force_offload_none": False,
        "fp32_audio": False,
        "guard_cuda_sync": False,
    }

def test_ltx_mps_gating_steers_apple_silicon_to_mps_recipe():
    gating = ltx_mps_gating(cuda_available=False, device_str="mps")
    assert gating == {
        "device": "mps",
        "disable_fp8": True,
        "force_offload_none": True,
        "fp32_audio": True,
        "guard_cuda_sync": True,
    }

def test_ltx_mps_gating_disables_cuda_features_on_cpu_without_forcing_mps_device():
    # A CPU host off CUDA still must drop fp8/offload (both CUDA-only) and guard the
    # unguarded cuda.synchronize, but it must not claim an mps device it does not have.
    gating = ltx_mps_gating(cuda_available=False, device_str="cpu")
    assert gating["device"] is None
    assert gating["disable_fp8"] is True
    assert gating["force_offload_none"] is True
    assert gating["guard_cuda_sync"] is True

def test_friendly_failure_identifies_ltx_frame_count_errors():
    message, error = friendly_failure("Video generation", RuntimeError("num_frames must be divisible by 8 + 1"))

    assert message == "Video generation failed because LTX requires a compatible frame count."
    assert "(frames - 1)" in error
    assert "Technical detail" in error

def test_video_job_reports_dynamic_loaded_models_on_progress_and_keepalive(monkeypatch):
    heartbeat_models = []
    blocking_models = []

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                heartbeat_models.append(payload["loadedModels"])
                return {}
            if path.endswith("/progress"):
                return {"status": payload["status"], "stage": payload["stage"]}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def __init__(self):
            self.models = []

        def loaded_models(self):
            return list(self.models)

        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            self.models = ["video-model-loaded"]

        def estimate_requirements(self, _request):
            return {"previewFrames": 1}

        def run(self, *, settings, job, request, progress, cancel_requested):
            self.models = ["video-model-running"]
            progress("running", "generating", 0.5, "Rendering.")
            return {"assetId": "asset-video-1"}

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            raise AssertionError("cleanup should not be called")

    def run_immediately(_api, _settings, _job_id, _status, callback, *, loaded_models, on_force_terminate=None, peaks=None):
        blocking_models.append(loaded_models())
        result = callback(lambda: False)
        blocking_models.append(loaded_models())
        return result

    monkeypatch.setattr("scene_worker.runtime.create_video_adapter", lambda _job=None: VideoAdapter())
    monkeypatch.setattr("scene_worker.runtime.run_blocking_job_step", run_immediately)

    run_video_job(
        Api(),
        SimpleNamespace(worker_id="worker-1"),
        {"id": "job-1", "payload": {"projectId": "project-1", "prompt": "clip"}},
    )

    assert heartbeat_models == [
        [],
        [],
        ["video-model-loaded"],
        ["video-model-running"],
        ["video-model-running"],
    ]
    assert blocking_models == [["video-model-loaded"], ["video-model-running"]]

def test_video_job_estimate_progress_accepts_non_preview_frame_requirements(monkeypatch):
    progress_messages = []

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                return {}
            if path.endswith("/progress"):
                progress_messages.append(payload["message"])
                return {"status": payload["status"], "stage": payload["stage"]}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            return None

        def estimate_requirements(self, _request):
            return {"estimatedFrames": 121, "requestedFrames": 120}

        def run(self, *, settings, job, request, progress, cancel_requested):
            return {"assetId": "asset-video-1"}

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            raise AssertionError("cleanup should not be called")

    monkeypatch.setattr("scene_worker.runtime.create_video_adapter", lambda _job=None: VideoAdapter())
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda *_args, **_kwargs: _args[4](lambda: False),
    )

    run_video_job(
        Api(),
        SimpleNamespace(worker_id="worker-1"),
        {"id": "job-1", "payload": {"projectId": "project-1", "prompt": "clip"}},
    )

    assert "Estimated 121 frames for this clip." in progress_messages

def test_video_job_failure_runs_cleanup_to_free_gpu(monkeypatch):
    events = {"cleanup": 0, "status": None}

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                return {}
            if path.endswith("/progress"):
                events["status"] = payload.get("status")
                return {"status": payload["status"], "stage": payload.get("stage")}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            return None

        def estimate_requirements(self, _request):
            return {"estimatedFrames": 121, "requestedFrames": 120}

        def run(self, *, settings, job, request, progress, cancel_requested):
            raise RuntimeError("CUDA error: out of memory")

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            events["cleanup"] += 1

    monkeypatch.setattr("scene_worker.runtime.create_video_adapter", lambda _job=None: VideoAdapter())
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda *_args, **_kwargs: _args[4](lambda: False),
    )

    # A CUDA OOM cleans up, marks the job failed, then exits (SystemExit) so the
    # supervisor restarts the child with a fresh CUDA context — the poisoned
    # context can't reliably reclaim VRAM in place.
    with pytest.raises(SystemExit):
        run_video_job(
            Api(),
            SimpleNamespace(worker_id="worker-1", gpu_id="0"),
            {"id": "job-oom", "payload": {"projectId": "project-1", "prompt": "clip"}},
        )

    assert events["cleanup"] == 1
    assert events["status"] == "failed"

def test_upscale_and_detail_jobs_restart_after_cuda_oom(monkeypatch):
    # sc-4187: upscale/detail set needs_oom_restart but never restarted — the
    # worker kept running with a poisoned CUDA context. They now release the
    # activation pool and restart like every other GPU handler.
    from scene_worker.runtime import run_detail_job, run_upscale_job

    class Api:
        def __init__(self):
            self.status = None

        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                return {}
            if path.endswith("/progress"):
                self.status = payload.get("status")
                return {"status": payload["status"], "stage": payload.get("stage")}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    for handler, runner_target in (
        (run_upscale_job, "scene_worker.runtime.run_image_upscale"),
        (run_detail_job, "scene_worker.runtime.run_image_detail"),
    ):
        released = {"count": 0}

        def boom(*_args, **_kwargs):
            raise RuntimeError("CUDA error: out of memory")

        monkeypatch.setattr(runner_target, boom)
        monkeypatch.setattr("scene_worker.runtime.find_project_path", lambda *_a, **_k: None)
        monkeypatch.setattr(
            "scene_worker.runtime.release_image_worker_memory",
            lambda: released.__setitem__("count", released["count"] + 1),
        )
        monkeypatch.setattr(
            "scene_worker.runtime.run_blocking_job_step",
            lambda *_args, **_kwargs: _args[4](lambda: False),
        )
        api = Api()
        with pytest.raises(SystemExit):
            handler(
                api,
                SimpleNamespace(worker_id="worker-1", gpu_id="0", data_dir=Path(".")),
                {"id": "job-oom", "payload": {"projectId": "project-1"}},
            )
        assert api.status == "failed"
        assert released["count"] == 1, handler.__name__

def test_video_job_nonoom_failure_does_not_restart(monkeypatch):
    events = {"cleanup": 0, "status": None}

    class Api:
        def post(self, path, payload):
            if path.endswith("/heartbeat"):
                return {}
            if path.endswith("/progress"):
                events["status"] = payload.get("status")
                return {"status": payload["status"], "stage": payload.get("stage")}
            raise AssertionError(path)

        def get(self, _path):
            return {"cancelRequested": False}

    class VideoAdapter:
        def prepare(self, *, settings, job):
            return {"job": job["id"]}

        def ensure_models(self, _request):
            return None

        def estimate_requirements(self, _request):
            return {"estimatedFrames": 121, "requestedFrames": 120}

        def run(self, *, settings, job, request, progress, cancel_requested):
            raise RuntimeError("num_frames must be divisible by 8 + 1")

        def cancel(self, _job_id):
            raise AssertionError("cancel should not be called")

        def cleanup(self, _job_id):
            events["cleanup"] += 1

    monkeypatch.setattr("scene_worker.runtime.create_video_adapter", lambda _job=None: VideoAdapter())
    monkeypatch.setattr(
        "scene_worker.runtime.run_blocking_job_step",
        lambda *_args, **_kwargs: _args[4](lambda: False),
    )

    # A non-OOM failure cleans up and marks failed but returns normally (no restart).
    run_video_job(
        Api(),
        SimpleNamespace(worker_id="worker-1", gpu_id="0"),
        {"id": "job-fail", "payload": {"projectId": "project-1", "prompt": "clip"}},
    )

    assert events["cleanup"] == 1
    assert events["status"] == "failed"

def test_random_batch_seeds_are_used_per_image():
    assert resolve_seed(None, "city at night", 2, [101, 202, 303, 404]) == 303

def test_explicit_seed_uses_reproducible_ladder():
    assert resolve_seed(1234, "city at night", 2, [101, 202, 303, 404]) == 1236

def test_video_adapter_override_aliases_and_unknown_values(monkeypatch):
    monkeypatch.delenv("SCENEWORKS_VIDEO_ADAPTER", raising=False)
    # Epic 3018 cutover: the Python worker no longer routes video to MLX (Wan/LTX
    # MLX-eligible jobs are claimed by the Rust GPU worker), so auto-dispatch always
    # lands on the torch adapters by model target.
    assert create_video_adapter({"payload": {"model": "ltx_2_3"}}).__class__.__name__ == "LtxPipelinesVideoAdapter"
    assert create_video_adapter({"payload": {"model": "wan_2_2"}}).__class__.__name__ == "DiffusersVideoAdapter"
    assert create_video_adapter({"payload": {"model": "wan_2_2_t2v_14b"}}).__class__.__name__ == "DiffusersVideoAdapter"
    assert create_video_adapter({"payload": {"model": "wan_2_2_i2v_14b"}}).__class__.__name__ == "DiffusersVideoAdapter"
    assert create_video_adapter().__class__.__name__ == "LtxPipelinesVideoAdapter"

    monkeypatch.setenv("SCENEWORKS_VIDEO_ADAPTER", "procedural")
    assert create_video_adapter().__class__.__name__ == "ProceduralVideoAdapter"

    monkeypatch.setenv("SCENEWORKS_VIDEO_ADAPTER", "ltx_pipelines")
    assert create_video_adapter().__class__.__name__ == "LtxPipelinesVideoAdapter"

    monkeypatch.setenv("SCENEWORKS_VIDEO_ADAPTER", "diffusers_video")
    assert create_video_adapter().__class__.__name__ == "DiffusersVideoAdapter"

    monkeypatch.setenv("SCENEWORKS_VIDEO_ADAPTER", "typo")
    try:
        create_video_adapter()
    except RuntimeError as exc:
        assert "Unsupported SCENEWORKS_VIDEO_ADAPTER" in str(exc)
    else:
        raise AssertionError("Unknown video adapter override should fail loudly.")

def test_create_video_adapter_routes_svd_to_diffusers(monkeypatch):
    # SVD is a diffusers pipeline (not the native LTX stack), so it routes to the
    # generic DiffusersVideoAdapter.
    monkeypatch.delenv("SCENEWORKS_VIDEO_ADAPTER", raising=False)
    adapter = create_video_adapter({"payload": {"model": "svd", "mode": "image_to_video"}})
    assert adapter.__class__.__name__ == "DiffusersVideoAdapter"

def test_svd_video_model_target_defaults():
    target = VIDEO_MODEL_TARGETS["svd"]
    assert target["adapter"] == "svd_video"
    assert target["family"] == "svd"
    # Image-conditioned only — no text_to_video or timeline modes.
    assert target["capabilities"] == ["image_to_video"]
    assert target["repo"] == "stabilityai/stable-video-diffusion-img2vid-xt"
    assert target["variant"] == "fp16"
    # Fixed-length clip defined by the checkpoint.
    assert target["numFrames"] == 25

def test_svd_num_frames_is_fixed_regardless_of_duration():
    adapter = DiffusersVideoAdapter()
    # Duration/fps would imply 60 frames for a 6s clip, but SVD emits its fixed
    # 25-frame burst regardless.
    request = video_request_from_job(
        {"id": "job-svd", "payload": {"projectId": "p", "mode": "image_to_video", "model": "svd", "duration": 6, "fps": 10}}
    )
    assert adapter._num_frames(request) == 25

def test_svd_pipeline_class_resolves_stable_video_diffusion():
    adapter = DiffusersVideoAdapter()
    target = VIDEO_MODEL_TARGETS["svd"]
    request = video_request_from_job({"id": "j", "payload": {"projectId": "p", "mode": "image_to_video", "model": "svd"}})
    fake_diffusers = SimpleNamespace(StableVideoDiffusionPipeline="SVD_PIPE_CLASS")
    assert adapter._pipeline_class(fake_diffusers, request, target) == "SVD_PIPE_CLASS"
    # Missing pipeline class fails loudly rather than silently mis-routing.
    try:
        adapter._pipeline_class(SimpleNamespace(), request, target)
    except RuntimeError as exc:
        assert "StableVideoDiffusionPipeline" in str(exc)
    else:
        raise AssertionError("SVD must require StableVideoDiffusionPipeline.")

def test_svd_pipeline_kwargs_build_image_conditioning_without_prompt(monkeypatch):
    # The SVD branch of _pipeline_kwargs animates the source image with motion
    # controls and passes NO prompt/height/width/guidance (unlike Wan/LTX).
    adapter = DiffusersVideoAdapter()
    target = VIDEO_MODEL_TARGETS["svd"]

    class FakeGen:
        def manual_seed(self, _seed):
            return self

    fake_torch = SimpleNamespace(Generator=lambda _device: FakeGen())
    monkeypatch.setattr(
        "scene_worker.video_adapters.importlib.import_module",
        lambda name: fake_torch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr("scene_worker.video_adapters.select_torch_device", lambda *_a, **_k: "cpu")

    class FakeImage:
        def resize(self, _size):
            return "RESIZED_IMAGE"

    class FakePipe:
        # filter_call_kwargs keeps only named params, so this signature gates
        # which kwargs survive — proving the SVD branch produces these and no prompt.
        def __call__(
            self,
            *,
            image=None,
            num_frames=None,
            num_inference_steps=None,
            decode_chunk_size=None,
            motion_bucket_id=None,
            fps=None,
            noise_aug_strength=None,
            generator=None,
        ):
            return None

    request = video_request_from_job(
        {
            "id": "job-svd",
            "payload": {
                "projectId": "p",
                "mode": "image_to_video",
                "model": "svd",
                "advanced": {"motionBucketId": 90, "conditioningFps": 6},
            },
        }
    )
    kwargs = adapter._pipeline_kwargs(
        pipe=FakePipe(),
        project_path=Path("/tmp/project"),
        request=request,
        target=target,
        first_image=FakeImage(),
        last_image=None,
        seed=7,
        num_frames=25,
    )
    assert kwargs["image"] == "RESIZED_IMAGE"
    assert kwargs["num_frames"] == 25
    assert kwargs["motion_bucket_id"] == 90
    assert kwargs["fps"] == 6
    assert kwargs["noise_aug_strength"] == 0.02
    assert "prompt" not in kwargs
    assert "guidance_scale" not in kwargs

def test_diffusers_wan_gguf_quant_variant_selection():
    adapter = DiffusersVideoAdapter()
    # Explicit selection wins regardless of platform default.
    explicit = adapter._gguf_quant_variant(_wan_quant_request({"quantization": "gguf-q4_k_m"}), "mps")
    assert explicit["id"] == "gguf-q4_k_m"
    assert explicit["transformerFile"].endswith("HighNoise-Q4_K_M.gguf")
    assert explicit["transformer2File"].endswith("LowNoise-Q4_K_M.gguf")
    # Auto / empty falls back to the per-platform default: Q8_0 on MPS, Q4_K_M on CUDA.
    assert adapter._gguf_quant_variant(_wan_quant_request({}), "mps")["id"] == "gguf-q8_0"
    assert adapter._gguf_quant_variant(_wan_quant_request({"quantization": "auto"}), "cuda")["id"] == "gguf-q4_k_m"
    # No default for the platform (cpu) -> unquantized.
    assert adapter._gguf_quant_variant(_wan_quant_request({}), "cpu") is None
    # Explicit opt-out keywords -> unquantized.
    assert adapter._gguf_quant_variant(_wan_quant_request({"quantization": "none"}), "mps") is None
    assert adapter._gguf_quant_variant(_wan_quant_request({"quantization": "full"}), "cuda") is None
    # No quantization block in the manifest entry -> unquantized.
    assert adapter._gguf_quant_variant(_wan_quant_request({}, manifest={}), "mps") is None

def test_diffusers_wan_gguf_injects_high_and_low_experts(monkeypatch):
    adapter = DiffusersVideoAdapter()
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeTransformer:
        @staticmethod
        def from_single_file(path, **kwargs):
            calls.append((path, kwargs))
            return f"T[{path}]"

    fake_diffusers = SimpleNamespace(
        WanTransformer3DModel=FakeTransformer,
        GGUFQuantizationConfig=lambda **kwargs: ("QCFG", kwargs),
    )
    monkeypatch.setattr(adapter, "_resolve_gguf_file", lambda repo, file_name: f"/cache/{repo}/{file_name}")

    # A14B: both experts injected (high -> transformer, low -> transformer_2).
    kwargs: dict[str, Any] = {}
    variant = {
        "id": "gguf-q8_0",
        "format": "gguf",
        "repo": "R",
        "transformerFile": "hi.gguf",
        "transformer2File": "lo.gguf",
    }
    adapter._inject_gguf_experts(fake_diffusers, kwargs, "diffusers/repo", variant, "DT")
    assert kwargs["transformer"] == "T[/cache/R/hi.gguf]"
    assert kwargs["transformer_2"] == "T[/cache/R/lo.gguf]"
    # The diffusers repo is the config source; compute dtype threads to from_single_file.
    assert calls[0][1]["config"] == "diffusers/repo"
    assert calls[0][1]["torch_dtype"] == "DT"

    # 5B dense (single transformer): no transformer_2.
    dense_kwargs: dict[str, Any] = {}
    adapter._inject_gguf_experts(
        fake_diffusers, dense_kwargs, "diffusers/repo", {"format": "gguf", "repo": "R", "transformerFile": "only.gguf"}, "DT"
    )
    assert dense_kwargs["transformer"] == "T[/cache/R/only.gguf]"
    assert "transformer_2" not in dense_kwargs

    # Missing diffusers classes fail loudly rather than silently skipping quantization.
    with pytest.raises(RuntimeError, match="GGUF support"):
        adapter._inject_gguf_experts(SimpleNamespace(), {}, "repo", variant, "DT")

def test_video_pipeline_evicts_previous_pipeline_and_loaded_models():
    adapter = DiffusersVideoAdapter()
    adapter._pipeline = object()
    adapter._pipeline_key_value = "old"
    adapter._loaded_models = {"old-model"}

    class Torch:
        class cuda:
            emptied = False

            @classmethod
            def is_available(cls):
                return True

            @classmethod
            def empty_cache(cls):
                cls.emptied = True

    adapter._evict_pipeline(Torch)

    assert adapter._pipeline is None
    assert adapter._pipeline_key_value is None
    assert adapter.loaded_models() == []
    assert Torch.cuda.emptied is True

def test_ltx_frame_count_uses_nearest_8n_plus_one_value():
    assert ltx_frame_count(100) == 97
    assert ltx_frame_count(150) == 153
    assert ltx_frame_count(200) == 201
    assert ltx_frame_count(250) == 249

def test_ltx_video_requirements_report_normalized_frame_count():
    adapter = DiffusersVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "duration": 6,
                "fps": 25,
                "advanced": {},
            },
        }
    )

    requirements = adapter.estimate_requirements(request)

    assert requirements["requestedFrames"] == 150
    assert requirements["estimatedFrames"] == 153
    assert requirements["repo"] == "Lightricks/LTX-2.3"

def test_ltx_pipelines_multigpu_compat_installs_missing_type_module(monkeypatch):
    for module_name in (
        "ltx_pipelines",
        "ltx_pipelines.multigpu",
        "ltx_pipelines.multigpu.delegating_builder",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    parent = ModuleType("ltx_pipelines")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "ltx_pipelines", parent)

    install_ltx_pipelines_multigpu_compat()

    module = importlib.import_module("ltx_pipelines.multigpu.delegating_builder")
    with pytest.raises(RuntimeError, match="optional multigpu DelegatingBuilder"):
        module.DelegatingBuilder()

def test_require_patch_target_returns_existing_symbol():
    owner = SimpleNamespace(load_weights=lambda self: None)
    target = _require_patch_target(
        owner, "load_weights", pin="some-dep==1.0", patch="example patch"
    )
    assert target is owner.load_weights

def test_require_patch_target_raises_naming_pin_on_missing_symbol():
    owner = SimpleNamespace()
    with pytest.raises(VendorPatchDriftError) as excinfo:
        _require_patch_target(
            owner, "load_weights", pin="mlx-video-with-audio>=0.1.36,<0.2", patch="LTX LoRA wrap (sc-1647)"
        )
    message = str(excinfo.value)
    assert "load_weights" in message
    assert "mlx-video-with-audio>=0.1.36,<0.2" in message
    assert "LTX LoRA wrap (sc-1647)" in message

def test_require_patch_target_raises_when_required_callable_is_not_callable():
    owner = SimpleNamespace(load_weights="not-callable")
    with pytest.raises(VendorPatchDriftError, match="no longer callable"):
        _require_patch_target(
            owner, "load_weights", pin="some-dep==1.0", patch="example patch", require_callable=True
        )

def test_require_patch_target_allows_non_callable_when_not_required():
    owner = SimpleNamespace(some_attr=123)
    assert _require_patch_target(owner, "some_attr", pin="some-dep==1.0", patch="example patch") == 123

def test_video_adapter_tracks_and_discards_temp_outputs(tmp_path):
    # The temp registry lives on the base VideoGenerationAdapter, so the Diffusers
    # adapter (which extends the base directly) gets force-cancel reaping too via the
    # already-wired on_force_terminate hook (sc-1719).
    adapter = DiffusersVideoAdapter()
    first = tmp_path / "a.tmp.mp4"
    second = tmp_path / "b.control.mp4"
    first.write_bytes(b"x")
    second.write_bytes(b"y")
    adapter.track_temp_output("job-1", first)
    adapter.track_temp_output("job-1", second)

    adapter.discard_temp_outputs("job-1")

    assert not first.exists()
    assert not second.exists()
    # Idempotent: the entry is popped, so a later cleanup after the force-cancel hook
    # is a harmless no-op.
    adapter.discard_temp_outputs("job-1")

def test_diffusers_cleanup_reaps_temp_outputs(tmp_path):
    adapter = DiffusersVideoAdapter()
    temp = tmp_path / "clip.tmp.mp4"
    temp.write_bytes(b"x")
    adapter.track_temp_output("job-4", temp)

    adapter.cleanup("job-4")

    assert not temp.exists()

def test_lens_adapter_discards_sidecar_scratch_dir(tmp_path):
    adapter = LensTurboAdapter()
    scratch = tmp_path / "lens_sidecar_abc"
    scratch.mkdir()
    (scratch / "spec.json").write_text("{}", encoding="utf-8")
    adapter._scratch_dir = scratch

    adapter.discard_temp_outputs("job-5")

    assert not scratch.exists()
    assert adapter._scratch_dir is None
    # No scratch dir registered -> no-op.
    adapter.discard_temp_outputs("job-5")

def test_native_ltx_adapter_reports_mocked_pipeline_requirements(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "duration": 6,
                "fps": 25,
                "quality": "fast",
                "advanced": {"mockNativeInference": True},
            },
        },
    )

    adapter.ensure_models(request)
    requirements = adapter.estimate_requirements(request)

    assert requirements["adapter"] == "ltx_pipelines"
    assert requirements["pipeline"] == "ltx_pipelines.distilled"
    assert requirements["requestedFrames"] == 150
    assert requirements["estimatedFrames"] == 153
    assert requirements["mockedInference"] is True
    assert requirements["resources"]["checkpointPath"] == str(checkpoint)

def test_native_ltx_pipeline_override_decouples_from_quality(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)

    def pipeline_for(quality, advanced):
        adapter = LtxPipelinesVideoAdapter()
        request = adapter.prepare(
            settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
            job={
                "id": "job-override",
                "payload": {
                    "projectId": "project-1",
                    "mode": "text_to_video",
                    "prompt": "city",
                    "model": "ltx_2_3",
                    "modelManifestEntry": manifest_entry,
                    "duration": 6,
                    "fps": 25,
                    "quality": quality,
                    "advanced": {"mockNativeInference": True, **advanced},
                },
            },
        )
        adapter.ensure_models(request)
        return adapter.estimate_requirements(request)["pipeline"]

    # Distilled override forces single-stage even at balanced quality.
    assert pipeline_for("balanced", {"ltxPipeline": "distilled"}) == "ltx_pipelines.distilled"
    # Two-stage override forces the dev + upscaler path even at fast quality.
    assert pipeline_for("fast", {"ltxPipeline": "two_stage"}) == "ltx_pipelines.ti2vid_two_stages"
    # Auto preserves the quality-driven default.
    assert pipeline_for("balanced", {"ltxPipeline": "auto"}) == "ltx_pipelines.ti2vid_two_stages"
    assert pipeline_for("fast", {}) == "ltx_pipelines.distilled"

def test_native_ltx_precision_selects_quantization_and_offload(monkeypatch):
    import scene_worker.video_adapters as va

    fake_offload = SimpleNamespace(CPU="cpu", DISK="disk", NONE="none")

    def fake_import(name):
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=fake_offload)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.install_ltx_pipelines_multigpu_compat", lambda: None)
    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import)
    adapter = va.LtxPipelinesVideoAdapter()

    def req(advanced):
        return SimpleNamespace(advanced=advanced)

    # Default precision is fp8 -> a quantization policy is built.
    assert adapter._quantization(req({})) == "fp8-cast"
    # Explicit bf16 -> no quantization.
    assert adapter._quantization(req({"precision": "bf16"})) is None
    # Default offload is resident ("none") regardless of precision; CPU streaming
    # leaks/thrashes on this stack, so callers opt into it explicitly.
    assert adapter._offload_mode(req({"precision": "bf16"})) == "none"
    assert adapter._offload_mode(req({})) == "none"
    assert adapter._offload_mode(req({"offloadMode": "cpu"})) == "cpu"
    # The override (used for the torch.compile path) forces resident.
    assert adapter._offload_mode(req({"offloadMode": "cpu"}), override="none") == "none"

def test_native_ltx_distilled_variant_switches_files(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    manifest_dir = config_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    resources = {
        "checkpoint": {"repo": "Lightricks/LTX-2.3", "file": "ltx-2.3-22b-dev.safetensors"},
        "distilledCheckpoint": {
            "repo": "Lightricks/LTX-2.3",
            "file": "ltx-2.3-22b-distilled-1.1.safetensors",
            "variants": {
                "1.1": "ltx-2.3-22b-distilled-1.1.safetensors",
                "1.0": "ltx-2.3-22b-distilled.safetensors",
            },
        },
        "spatialUpscaler": {"repo": "Lightricks/LTX-2.3", "file": "spatial.safetensors"},
        "distilledLora": {
            "repo": "Lightricks/LTX-2.3",
            "file": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "variants": {
                "1.1": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                "1.0": "ltx-2.3-22b-distilled-lora-384.safetensors",
            },
        },
        "gemma": {"repo": "google/gemma-3-12b-it-qat-q4_0-unquantized"},
    }
    manifest_entry = {
        "id": "ltx_2_3",
        "name": "LTX-2.3",
        "family": "ltx-video",
        "type": "video",
        "adapter": "ltx_video",
        "capabilities": ["text_to_video"],
        "downloads": [],
        "paths": {},
        "resources": resources,
        "defaults": {},
        "limits": {},
        "loraCompatibility": {},
        "ui": {},
    }
    (manifest_dir / "builtin.models.jsonc").write_text(
        json.dumps({"schemaVersion": 1, "models": [manifest_entry]}),
        encoding="utf-8",
    )

    def resolve(quality, advanced):
        adapter = LtxPipelinesVideoAdapter()
        request = adapter.prepare(
            settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
            job={
                "id": "job-variant",
                "payload": {
                    "projectId": "project-1",
                    "mode": "text_to_video",
                    "prompt": "city",
                    "model": "ltx_2_3",
                    "modelManifestEntry": manifest_entry,
                    "duration": 6,
                    "fps": 25,
                    "quality": quality,
                    "advanced": advanced,
                },
            },
        )
        return adapter.resolve_resources(request)

    # Single-stage distilled: the variant selects the checkpoint file.
    assert resolve("fast", {}).checkpoint_path.name == "ltx-2.3-22b-distilled-1.1.safetensors"
    assert resolve("fast", {"distilledVariant": "1.0"}).checkpoint_path.name == "ltx-2.3-22b-distilled.safetensors"
    # Two-stage: the variant selects the distilled LoRA file (dev checkpoint is unversioned).
    two_stage = resolve("balanced", {"distilledVariant": "1.0"})
    assert two_stage.checkpoint_path.name == "ltx-2.3-22b-dev.safetensors"
    assert two_stage.distilled_lora_path.name == "ltx-2.3-22b-distilled-lora-384.safetensors"

def test_native_ltx_missing_resources_reports_all_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-home"))
    manifest_entry = write_native_ltx_manifest(config_dir)
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "advanced": {},
            },
        },
    )

    with pytest.raises(RuntimeError) as exc:
        adapter.ensure_models(request)

    message = str(exc.value)
    assert "checkpointPath" in message
    assert "spatialUpscalerPath" in message
    assert "distilledLoraPath" in message
    assert "gemmaRoot" in message
    assert str(data_dir / "models" / safe_download_dir("Lightricks/LTX-2.3") / "checkpoint.safetensors") in message
    assert str(data_dir / "models" / safe_download_dir("google/gemma-3-12b-it-qat-q4_0-unquantized")) in message

def test_native_ltx_resources_resolve_from_huggingface_cache(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    cache_root = tmp_path / "hf" / "hub"
    data_dir.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache_root))
    manifest_entry = write_native_ltx_manifest(config_dir)
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "checkpoint.safetensors")
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "spatial.safetensors")
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "distilled-lora.safetensors")
    gemma_snapshot = write_huggingface_cache_resource(cache_root, "google/gemma-3-12b-it-qat-q4_0-unquantized", "config.json")
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "advanced": {"mockNativeInference": True},
            },
        },
    )

    adapter.ensure_models(request)
    resources = adapter.estimate_requirements(request)["resources"]

    assert resources["checkpointPath"].endswith("checkpoint.safetensors")
    assert str(cache_root) in resources["checkpointPath"]
    assert resources["spatialUpscalerPath"].endswith("spatial.safetensors")
    assert resources["distilledLoraPath"].endswith("distilled-lora.safetensors")
    assert resources["gemmaRoot"] == str(gemma_snapshot)

def test_native_ltx_resources_resolve_from_mounted_data_cache_without_hf_env(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    cache_root = data_dir / "cache" / "huggingface" / "hub"
    data_dir.mkdir()
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    manifest_entry = write_native_ltx_manifest(config_dir)
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "checkpoint.safetensors")
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "spatial.safetensors")
    write_huggingface_cache_resource(cache_root, "Lightricks/LTX-2.3", "distilled-lora.safetensors")
    gemma_snapshot = write_huggingface_cache_resource(cache_root, "google/gemma-3-12b-it-qat-q4_0-unquantized", "config.json")
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "advanced": {"mockNativeInference": True},
            },
        },
    )

    adapter.ensure_models(request)
    resources = adapter.estimate_requirements(request)["resources"]

    assert resources["spatialUpscalerPath"].startswith(str(cache_root))
    assert resources["distilledLoraPath"].startswith(str(cache_root))
    assert resources["gemmaRoot"] == str(gemma_snapshot)

def test_native_ltx_fast_pipeline_does_not_require_distilled_lora(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    checkpoint, spatial, _lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, gemma=gemma)
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "quality": "fast",
                "advanced": {"mockNativeInference": True},
            },
        },
    )

    adapter.ensure_models(request)
    requirements = adapter.estimate_requirements(request)

    assert requirements["pipeline"] == "ltx_pipelines.distilled"

def test_native_ltx_advanced_resource_overrides_win(tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    write_native_ltx_manifest(config_dir)
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "advanced": {
                    "mockNativeInference": True,
                    "checkpointPath": str(checkpoint),
                    "spatialUpscalerPath": str(spatial),
                    "distilledLoraPath": str(lora),
                    "gemmaRoot": str(gemma),
                },
            },
        },
    )

    adapter.ensure_models(request)
    resources = adapter.estimate_requirements(request)["resources"]

    assert resources["checkpointPath"] == str(checkpoint)
    assert resources["spatialUpscalerPath"] == str(spatial)
    assert resources["distilledLoraPath"] == str(lora)
    assert resources["gemmaRoot"] == str(gemma)

def test_native_ltx_adapter_rejects_unsupported_modes():
    adapter = LtxPipelinesVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "edit_image",
                "prompt": "city",
                "model": "ltx_2_3",
                "advanced": {},
            },
        }
    )

    with pytest.raises(RuntimeError, match="native pipelines currently support"):
        adapter.ensure_models(request)

def test_native_ltx_mocked_run_writes_scene_video_asset(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-ltx",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "Neon harbor",
            "model": "ltx_2_3",
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "advanced": {"mockNativeInference": True},
        },
    }
    monkeypatch.setattr(
        "scene_worker.video_adapters.gradient_frame",
        lambda width, height, _digest: Image.new("RGB", (width, height), "navy"),
    )
    adapter = LtxPipelinesVideoAdapter()
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir), job=job)

    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    asset = result["assetWrites"][0]
    media_path = project_path / asset["mediaPath"]
    assert media_path.exists()
    assert result["adapter"] == "ltx_pipelines"
    assert asset["adapter"] == "ltx_pipelines"
    assert asset["rawAdapterSettings"]["pipeline"] == "ltx_pipelines.ti2vid_two_stages"
    assert asset["rawAdapterSettings"]["mockedNativeInference"] is True
    assert adapter.loaded_models() == ["ltx_2_3", "ltx_pipelines.ti2vid_two_stages"]

def test_native_ltx_text_to_video_uses_ltx_pipeline_and_writes_mp4(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    calls = {"init": None, "run": None, "encode": None}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, **kwargs):
            calls["run"] = kwargs
            return ["video-chunk"], "audio-track"

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeGuiderParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_encode_video(**kwargs):
        calls["encode"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={"rename": "map"},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ti2vid_two_stages":
            return SimpleNamespace(TI2VidTwoStagesPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 2,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        if name == "ltx_core.components.guiders":
            return SimpleNamespace(MultiModalGuiderParams=FakeGuiderParams)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    # Pin the CUDA gating recipe so the fp8 default is exercised deterministically.
    # On a host with torch+MPS installed the gating would disable fp8 (it only
    # assumes CUDA when torch is absent, e.g. the CI parity job), so without this
    # the fp8-cast assertion below is host-dependent.
    monkeypatch.setattr(
        adapter,
        "_ltx_device_gating",
        lambda: {
            "device": None,
            "disable_fp8": False,
            "force_offload_none": False,
            "fp32_audio": False,
            "guard_cuda_sync": False,
        },
    )
    job = {
        "id": "job-real-ltx",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "Neon harbor",
            "negativePrompt": "rain",
            "model": "ltx_2_3",
            "modelManifestEntry": manifest_entry,
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "advanced": {"steps": 7, "distilledLoraStrength": 0.6},
        },
    }
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir), job=job)

    adapter.ensure_models(request)
    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    asset = result["assetWrites"][0]
    media_path = project_path / asset["mediaPath"]
    assert media_path.read_bytes() == b"mp4"
    assert calls["init"]["checkpoint_path"] == str(checkpoint)
    assert calls["init"]["distilled_lora"] == [(str(lora), 0.6, {"rename": "map"})]
    # Default precision is fp8 with no torch.compile: quantization is passed and the
    # default offload is resident ("none"); CPU streaming is opt-in.
    assert calls["init"]["quantization"] == "fp8-cast"
    assert calls["init"]["offload_mode"] == "none"
    assert calls["run"]["prompt"] == "Neon harbor"
    assert calls["run"]["negative_prompt"] == "rain"
    assert calls["run"]["num_inference_steps"] == 7
    assert calls["run"]["images"] == []
    assert calls["encode"]["video"] == ["video-chunk"]
    assert calls["encode"]["audio"] == "audio-track"
    assert calls["encode"]["video_chunks_number"] == 2
    assert asset["mimeType"] == "video/mp4"
    assert asset["rawAdapterSettings"]["realModelInference"] is True
    assert asset["rawAdapterSettings"]["mockedNativeInference"] is False
    assert result["requirements"]["mockedInference"] is False

def test_native_ltx_dependency_probe_only_imports_selected_pipeline(monkeypatch):
    imported = []

    def fake_import_module(name):
        imported.append(name)
        if name == "ltx_pipelines.ic_lora":
            raise ImportError(name)
        return SimpleNamespace()

    monkeypatch.setattr("scene_worker.video_adapters.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    monkeypatch.setattr("scene_worker.video_adapters.install_ltx_pipelines_multigpu_compat", lambda: None)

    adapter = LtxPipelinesVideoAdapter()
    text_request = video_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "Neon harbor",
                "model": "ltx_2_3",
                "quality": "balanced",
            }
        }
    )
    ic_request = video_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "image_to_video",
                "prompt": "Neon harbor",
                "model": "ltx_2_3",
                "loras": [{"id": "identity", "icLora": True}],
            }
        }
    )

    assert adapter._dependencies_available(text_request) is True
    assert "ltx_pipelines.ti2vid_two_stages" in imported
    assert "ltx_pipelines.ic_lora" not in imported
    assert adapter._dependencies_available(ic_request) is False

def test_native_ltx_image_to_video_passes_source_image_conditioning(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    image_rel = "assets/images/source.png"
    (project_path / "assets" / "images").mkdir(parents=True)
    Image.new("RGB", (16, 16), "teal").save(project_path / image_rel)
    (project_path / "assets" / "images" / "source.sceneworks.json").write_text(
        json.dumps({"id": "asset-source", "file": {"path": image_rel}}),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    ic_lora = tmp_path / "identity-control.safetensors"
    ic_lora.write_bytes(b"ic-lora")
    calls = {"run": None, "encode": None}

    class FakePipeline:
        def __init__(self, **_kwargs):
            return None

        def __call__(self, **kwargs):
            calls["run"] = kwargs
            return ["video-chunk"], None

    class FakeConditioningInput(NamedTuple):
        path: str
        frame_idx: int
        strength: float

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeGuiderParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_encode_video(**kwargs):
        calls["encode"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ic_lora":
            return SimpleNamespace(ICLoraPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 1,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        if name == "ltx_core.components.guiders":
            return SimpleNamespace(MultiModalGuiderParams=FakeGuiderParams)
        if name == "ltx_pipelines.utils.args":
            return SimpleNamespace(ImageConditioningInput=FakeConditioningInput)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    job = {
        "id": "job-i2v",
        "payload": {
            "projectId": "project-1",
            "mode": "image_to_video",
            "prompt": "Make the harbor move",
            "model": "ltx_2_3",
            "modelManifestEntry": manifest_entry,
            "sourceAssetId": "asset-source",
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "loras": [
                {
                    "id": "identity_ic",
                    "name": "Identity Control",
                    "icLora": True,
                    "installedPath": str(ic_lora),
                    "weight": 0.65,
                    "families": ["ltx-video"],
                }
            ],
            "advanced": {"imageConditioningStrength": 0.7},
        },
    }
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir), job=job)

    adapter.ensure_models(request)
    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    image_condition = calls["run"]["images"][0]
    assert calls["run"]["video_conditioning"] == []
    assert image_condition.path == str(project_path / image_rel)
    assert image_condition.frame_idx == 0
    assert image_condition.strength == 0.7
    assert result["assetWrites"][0]["sourceAssetId"] == "asset-source"
    assert result["assetWrites"][0]["rawAdapterSettings"]["realModelInference"] is True

def test_native_ltx_image_to_video_falls_back_without_ic_lora(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    image_rel = "assets/images/source.png"
    (project_path / "assets" / "images").mkdir(parents=True)
    Image.new("RGB", (16, 16), "teal").save(project_path / image_rel)
    (project_path / "assets" / "images" / "source.sceneworks.json").write_text(
        json.dumps({"id": "asset-source", "file": {"path": image_rel}}),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    style_lora = tmp_path / "cinematic-style.safetensors"
    style_lora.write_bytes(b"style-lora")
    calls = {"init": None, "run": None}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, **kwargs):
            calls["run"] = kwargs
            return ["video-chunk"], None

    class FakeConditioningInput(NamedTuple):
        path: str
        frame_idx: int
        strength: float

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeGuiderParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_encode_video(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={"rename": "map"},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ti2vid_two_stages":
            return SimpleNamespace(TI2VidTwoStagesPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 1,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        if name == "ltx_core.components.guiders":
            return SimpleNamespace(MultiModalGuiderParams=FakeGuiderParams)
        if name == "ltx_pipelines.utils.args":
            return SimpleNamespace(ImageConditioningInput=FakeConditioningInput)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    request = adapter.prepare(
        settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir),
        job={
            "id": "job-i2v-missing-lora",
            "payload": {
                "projectId": "project-1",
                "mode": "image_to_video",
                "prompt": "Make the harbor move",
                "model": "ltx_2_3",
                "modelManifestEntry": manifest_entry,
                "sourceAssetId": "asset-source",
                "duration": 1,
                "fps": 12,
                "width": 320,
                "height": 256,
                "quality": "balanced",
                "loras": [
                    {
                        "id": "cinematic_style",
                        "name": "Cinematic Style",
                        "installedPath": str(style_lora),
                        "weight": 0.55,
                        "families": ["ltx-video"],
                    }
                ],
                "advanced": {"imageConditioningStrength": 0.75},
            },
        },
    )

    adapter.ensure_models(request)
    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job={"id": "job-i2v-missing-lora"},
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    assert calls["init"]["checkpoint_path"] == str(checkpoint)
    assert calls["init"]["distilled_lora"] == [(str(lora), 0.8, {"rename": "map"})]
    assert calls["init"]["loras"] == ((str(style_lora), 0.55, {"rename": "map"}),)
    assert calls["run"]["images"] == [FakeConditioningInput(str(project_path / image_rel), 0, 0.75)]
    assert "video_conditioning" not in calls["run"]
    assert result["requirements"]["pipeline"] == "ltx_pipelines.ti2vid_two_stages"

def test_native_ltx_extend_clip_uses_ic_lora_video_conditioning(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    video_rel = "assets/videos/source.mp4"
    (project_path / "assets" / "videos").mkdir(parents=True)
    (project_path / video_rel).write_bytes(b"source-video")
    (project_path / "assets" / "videos" / "source.sceneworks.json").write_text(
        json.dumps({"id": "asset-source-video", "type": "video", "file": {"path": video_rel}}),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)
    ic_lora = tmp_path / "identity-control.safetensors"
    ic_lora.write_bytes(b"ic-lora")
    calls = {"init": None, "run": None, "encode": None}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def __call__(self, **kwargs):
            calls["run"] = kwargs
            return ["video-chunk"], None

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    def fake_encode_video(**kwargs):
        calls["encode"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={"rename": "map"},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ic_lora":
            return SimpleNamespace(ICLoraPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 1,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    job = {
        "id": "job-extend-ic",
        "payload": {
            "projectId": "project-1",
            "mode": "extend_clip",
            "prompt": "Keep the character walking",
            "model": "ltx_2_3",
            "modelManifestEntry": manifest_entry,
            "sourceClipAssetId": "asset-source-video",
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "loras": [
                {
                    "id": "identity_ic",
                    "name": "Identity Control",
                    "icLora": True,
                    "installedPath": str(ic_lora),
                    "weight": 0.7,
                    "families": ["ltx-video"],
                }
            ],
            "advanced": {"videoConditioningStrength": 0.85, "conditioningAttentionStrength": 0.9},
        },
    }
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir), job=job)

    adapter.ensure_models(request)
    result = adapter.run(
        settings=SimpleNamespace(data_dir=data_dir),
        job=job,
        request=request,
        progress=lambda *_args: None,
        cancel_requested=lambda: False,
    )

    assert calls["init"]["distilled_checkpoint_path"] == str(checkpoint)
    assert calls["init"]["loras"] == ((str(ic_lora), 0.7, {"rename": "map"}),)
    assert calls["run"]["images"] == []
    assert calls["run"]["video_conditioning"] == [(str(project_path / video_rel), 0.85)]
    assert calls["run"]["conditioning_attention_strength"] == 0.9
    assert result["requirements"]["pipeline"] == "ltx_pipelines.ic_lora"
    assert result["assetWrites"][0]["sourceClipAssetId"] == "asset-source-video"

def test_native_ltx_cleanup_deletes_temp_output_and_evicts_pipeline(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    checkpoint, spatial, lora, gemma = write_native_ltx_resource_files(tmp_path)
    manifest_entry = write_native_ltx_manifest(config_dir, checkpoint=checkpoint, spatial=spatial, lora=lora, gemma=gemma)

    class FakePipeline:
        def __init__(self, **_kwargs):
            return None

        def __call__(self, **_kwargs):
            return ["video-chunk"], None

    class FakeTilingConfig:
        @staticmethod
        def default():
            return "tiling-config"

    class FakeOffloadMode:
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeGuiderParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_encode_video(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"partial")
        raise RuntimeError("encoder failed")

    def fake_import_module(name):
        if name == "ltx_core.loader":
            return SimpleNamespace(
                LoraPathStrengthAndSDOps=lambda path, strength, sd_ops: (path, strength, sd_ops),
                LTXV_LORA_COMFY_RENAMING_MAP={},
            )
        if name == "ltx_pipelines.utils.types":
            return SimpleNamespace(OffloadMode=FakeOffloadMode)
        if name == "ltx_core.quantization":
            return SimpleNamespace(QuantizationPolicy=_FakeQuantizationPolicy)
        if name == "ltx_pipelines.ti2vid_two_stages":
            return SimpleNamespace(TI2VidTwoStagesPipeline=FakePipeline)
        if name == "ltx_core.model.video_vae":
            return SimpleNamespace(
                TilingConfig=FakeTilingConfig,
                get_video_chunks_number=lambda _frames, _tiling: 1,
            )
        if name == "ltx_pipelines.utils.media_io":
            return SimpleNamespace(encode_video=fake_encode_video)
        if name == "ltx_core.components.guiders":
            return SimpleNamespace(MultiModalGuiderParams=FakeGuiderParams)
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.video_adapters.importlib.import_module", fake_import_module)
    adapter = LtxPipelinesVideoAdapter()
    monkeypatch.setattr(adapter, "_dependencies_available", lambda *_args: True)
    job = {
        "id": "job-cleanup",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_video",
            "prompt": "Neon harbor",
            "model": "ltx_2_3",
            "modelManifestEntry": manifest_entry,
            "duration": 1,
            "fps": 12,
            "width": 320,
            "height": 256,
            "quality": "balanced",
            "advanced": {},
        },
    }
    request = adapter.prepare(settings=SimpleNamespace(data_dir=data_dir, config_dir=config_dir), job=job)

    adapter.ensure_models(request)
    with pytest.raises(RuntimeError, match="encoder failed"):
        adapter.run(
            settings=SimpleNamespace(data_dir=data_dir),
            job=job,
            request=request,
            progress=lambda *_args: None,
            cancel_requested=lambda: False,
        )
    assert list((project_path / "assets" / "videos").rglob("*.tmp.mp4"))

    adapter.cleanup(job["id"])

    assert list((project_path / "assets" / "videos").rglob("*.tmp.mp4")) == []
    assert adapter.loaded_models() == []
    assert adapter._pipeline is None

def test_ltx_video_text_to_video_default_repo_fails_before_diffusers_404():
    adapter = DiffusersVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "advanced": {},
            },
        }
    )

    with pytest.raises(RuntimeError) as exc:
        adapter.ensure_models(request)

    assert "LTX-2.3 text-to-video is supported by the model" in str(exc.value)
    assert "model_index.json" in str(exc.value)

def test_ltx_video_image_modes_keep_image_to_video_diffusers_repo():
    adapter = DiffusersVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "image_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "sourceAssetId": "asset-image",
                "advanced": {},
            },
        }
    )

    requirements = adapter.estimate_requirements(request)

    assert requirements["repo"] == "Lightricks/LTX-Video"

def test_ltx_video_model_repo_override_wins_over_mode_specific_repos():
    adapter = DiffusersVideoAdapter()
    request = video_request_from_job(
        {
            "id": "job-1",
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_video",
                "prompt": "city",
                "model": "ltx_2_3",
                "advanced": {"modelRepo": "owner/custom-ltx-diffusers"},
            },
        }
    )

    requirements = adapter.estimate_requirements(request)

    assert requirements["repo"] == "owner/custom-ltx-diffusers"
    adapter.ensure_models(request)

def test_evenly_spaced_indices_are_bounded():
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(1, 4) == [0, 0, 0, 0]

def test_frames_from_output_accepts_nested_frames():
    red = Image.new("RGB", (2, 2), "red")
    blue = Image.new("RGB", (2, 2), "blue")

    frames = frames_from_output(SimpleNamespace(frames=[[red, blue]]))

    assert len(frames) == 2
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)

def test_frames_from_output_scales_float_numpy_frames():
    np = pytest.importorskip("numpy")

    # Diffusers video pipelines default to output_type="np": a float32 array in
    # [0, 1] shaped (batch, frames, H, W, 3). PIL rejects float RGB data, so
    # frames_from_output must scale to uint8 instead of raising
    # "Cannot handle this data type: (1, 1, 3), <f4".
    array = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
    array[0, 0, :, :, 0] = 1.0  # first frame fully red

    frames = frames_from_output(SimpleNamespace(frames=array))

    assert len(frames) == 2
    assert frames[0].mode == "RGB"
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)

def test_load_seekable_image_frame_does_not_fallback_on_decompression_bomb(monkeypatch, tmp_path):
    path = tmp_path / "bomb.png"
    path.write_bytes(b"not really used")

    monkeypatch.setattr("scene_worker.video_adapters.Image.open", lambda _path: (_ for _ in ()).throw(Image.DecompressionBombError("too large")))
    monkeypatch.setattr(
        "scene_worker.video_adapters.load_seekable_video_frame",
        lambda _path, _timestamp: (_ for _ in ()).throw(AssertionError("video fallback should not run")),
    )

    assert load_seekable_image_frame(path, 0) is None

def test_person_track_masks_fail_without_track_boxes(tmp_path):
    project_path = tmp_path
    track_dir = project_path / "person-tracks"
    track_dir.mkdir()
    (track_dir / "track_empty.sceneworks.person-track.json").write_text(
        json.dumps({"id": "track_empty", "frames": [], "selectedDetection": {}}),
        encoding="utf-8",
    )

    try:
        person_track_masks(project_path, "track_empty", 64, 64, 2)
    except RuntimeError as exc:
        assert "no usable boxes" in str(exc)
    else:
        raise AssertionError("Empty person tracks should fail loudly.")

def test_character_reference_images_are_capped(tmp_path):
    project_path = tmp_path
    (project_path / "characters").mkdir()
    (project_path / "assets" / "images").mkdir(parents=True)
    references = []
    for index in range(5):
        asset_id = f"asset_ref_{index}"
        media_rel = f"assets/images/ref_{index}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(project_path / media_rel)
        (project_path / f"assets/images/ref_{index}.sceneworks.json").write_text(
            json.dumps({"id": asset_id, "file": {"path": media_rel}}),
            encoding="utf-8",
        )
        references.append({"assetId": asset_id, "approved": True})
    (project_path / "characters" / "character_1.sceneworks.character.json").write_text(
        json.dumps({"id": "character_1", "references": references, "looks": []}),
        encoding="utf-8",
    )

    assert len(character_reference_images(project_path, "character_1", None, 16, 16)) == 4

def test_replace_person_video_result_carries_lineage_facts():
    # sc-1656 slice 3: the worker reports flat video facts and the Rust API builds
    # the sidecar. Pin the facts the worker emits for a replace_person job — the
    # honest personDetection/replacement defaults now live in the Rust builder
    # (project_store::build_generated_asset_sidecar), tested Rust-side.
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "replace_person",
            "prompt": "Replace the hero",
            "model": "wan_2_2",
            "sourceClipAssetId": "asset-video",
            "personTrackId": "track-1",
            "characterId": "character-1",
            "characterLookId": "look-1",
            "replacementMode": "full_person_keep_outfit",
            "advanced": {},
        },
    }
    request = video_request_from_job(job)

    result = video_generation_result(
        request=request,
        target=VIDEO_MODEL_TARGETS["wan_2_2"],
        adapter_id="wan_video",
        asset_id="asset-output",
        generation_set_id="genset-1",
        media_rel="assets/videos/replacement.mp4",
        seed=44,
        created_at="2026-05-17T00:00:00Z",
        mime_type="video/mp4",
        raw_settings={},
    )
    fact = result["assetWrites"][0]
    assert fact["type"] == "video"
    assert fact["mode"] == "replace_person"
    assert fact["mimeType"] == "video/mp4"
    assert fact["personTrackId"] == "track-1"
    assert fact["replacementMode"] == "full_person_keep_outfit"
    assert fact["sourceClipAssetId"] == "asset-video"
    assert fact["characterId"] == "character-1"
    # No real masked-control path ran here, so the worker reports no
    # replacementStatus; the Rust builder fills the honest false defaults.
    assert "replacementStatus" not in fact
