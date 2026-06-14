from __future__ import annotations

from worker_runtime_shared import *

def _write_asset_sidecar(project_path, folder, asset_id, media_path):
    sidecar_path = project_path / folder / f"{asset_id}.sceneworks.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps({"id": asset_id, "file": {"path": media_path}}),
        encoding="utf-8",
    )
    return sidecar_path

def test_find_asset_media_path_rejects_sidecar_paths_outside_project(tmp_path):
    from scene_worker.image_adapters import find_asset_media_path

    in_project_media = tmp_path / "assets" / "images" / "source.png"
    in_project_media.parent.mkdir(parents=True, exist_ok=True)
    in_project_media.write_bytes(b"not-a-real-png")
    _write_asset_sidecar(tmp_path, "assets/images", "asset_inside", "assets/images/source.png")

    assert find_asset_media_path(tmp_path, "asset_inside") == in_project_media.resolve()

    outside_media = tmp_path.parent / f"{tmp_path.name}-outside.png"
    outside_media.write_bytes(b"outside")
    _write_asset_sidecar(tmp_path, "assets/images", "asset_absolute_escape", str(outside_media))
    _write_asset_sidecar(tmp_path, "assets/images", "asset_relative_escape", f"../{outside_media.name}")

    for asset_id in ("asset_absolute_escape", "asset_relative_escape"):
        with pytest.raises(RuntimeError, match="outside the project"):
            find_asset_media_path(tmp_path, asset_id)

def test_run_image_upscale_writes_single_child_asset(tmp_path, monkeypatch):
    # sc-2431: standalone upscale of an existing asset. Mock the engine so the
    # orchestration (source resolve → one child asset + lineage) is exercised
    # without torch/weights (CI has neither).
    from scene_worker.image_adapters import run_image_upscale

    source = Image.new("RGB", (8, 6), "blue")
    source_path = tmp_path / "src.png"
    source.save(source_path, "PNG")

    class _FakeUpscaler:
        id = "real-esrgan"

        def upscale(self, image, *, request, cancel_requested):
            factor = request.upscale.factor
            return image.resize((image.width * factor, image.height * factor))

    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: source_path,
    )
    monkeypatch.setattr(
        "scene_worker.image_adapters.create_image_upscaler",
        lambda request, **kwargs: _FakeUpscaler(),
    )

    job = {
        "id": "job_upscale_1",
        "payload": {
            "projectId": "project_1",
            "sourceAssetId": "asset_src",
            "factor": 2,
            "engine": "real-esrgan",
            "displayName": "Studio still",
        },
    }
    result = run_image_upscale(
        SimpleNamespace(gpu_id=None),
        job,
        project_path=tmp_path,
        progress=lambda *args: None,
        cancel_requested=lambda: False,
    )

    writes = result["assetWrites"]
    assert len(writes) == 1
    fact = writes[0]
    # Upscaled to 2x of the 8x6 source.
    assert (fact["width"], fact["height"]) == (16, 12)
    assert fact["mode"] == "image_upscale"
    assert fact["sourceAssetId"] == "asset_src"
    assert fact["parents"] == ["asset_src"]
    assert fact["displayName"] == "Studio still (2x upscaled)"
    assert fact["extra"] == {
        "isUpscaled": True,
        "upscaledFromAssetId": "asset_src",
        "factor": 2,
        "engine": "real-esrgan",
    }
    # The upscaled PNG was actually written to the reported path at 2x size.
    written_path = tmp_path / fact["mediaPath"]
    assert written_path.is_file()
    assert Image.open(written_path).size == (16, 12)
    # One generation set wrapping the single child asset.
    assert result["expectedCount"] == 1
    assert result["generationSet"]["id"] == result["generationSetId"]

def test_image_request_threads_mask_asset_id(monkeypatch):
    # sc-2476: an edit job can carry an inpaint mask alongside the source.
    request = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "maskAssetId": "asset_mask"}}
    )
    assert request.mask_asset_id == "asset_mask"
    # Absent maskAssetId → None (whole-image edit).
    plain = image_request_from_job({"payload": {"projectId": "p", "sourceAssetId": "asset_src"}})
    assert plain.mask_asset_id is None

def test_model_supports_inpaint_matrix():
    # sc-2476: SDXL family honors a mask; instruction-edit / img2img-only models don't.
    assert model_supports_inpaint("sdxl") is True
    assert model_supports_inpaint("realvisxl") is True
    assert model_supports_inpaint("qwen_image_edit_2511") is False
    assert model_supports_inpaint("z_image_edit") is False
    assert model_supports_inpaint("unknown_model") is False

def test_load_mask_image_resolves_grayscale_and_aligns_to_output(tmp_path, monkeypatch):
    # sc-2476: the mask resolves only through the project sidecar (no path escape) and
    # is returned grayscale, resized to the output W×H so it aligns with the source.
    mask = Image.new("RGB", (40, 30), "white")
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path, "PNG")
    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: mask_path,
    )
    request = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "maskAssetId": "asset_mask", "width": 512, "height": 256}}
    )
    loaded = load_mask_image(tmp_path, request)
    assert loaded.mode == "L"
    assert loaded.size == (512, 256)

    # No mask id → explicit error (callers gate on mask_asset_id first).
    with pytest.raises(RuntimeError, match="mask image asset"):
        load_mask_image(tmp_path, image_request_from_job({"payload": {"projectId": "p"}}))

def test_normalize_fit_mode_defaults_to_crop():
    # sc-2552: missing/unknown fit modes fall back to crop (never silently stretch).
    assert normalize_fit_mode(None) == "crop"
    assert normalize_fit_mode("") == "crop"
    assert normalize_fit_mode("bogus") == "crop"
    # Known modes pass through, case-insensitively.
    assert normalize_fit_mode("crop") == "crop"
    assert normalize_fit_mode("pad") == "pad"
    assert normalize_fit_mode("outpaint") == "outpaint"
    assert normalize_fit_mode("Stretch") == "stretch"
    # Default on the request model is crop when fitMode is absent.
    assert image_request_from_job({"payload": {"projectId": "p"}}).fit_mode == "crop"
    assert (
        image_request_from_job({"payload": {"projectId": "p", "fitMode": "pad"}}).fit_mode == "pad"
    )

def test_fit_image_crop_pad_stretch_geometry():
    # sc-2552: a 2:1 source fitted to a 1:1 box. crop covers (no bars), pad letterboxes
    # (neutral bars), stretch distorts. All return exactly the target size.
    src = Image.new("RGB", (40, 20), (255, 0, 0))  # solid red, landscape 2:1

    cropped = fit_image(src, 100, 100, "crop")
    assert cropped.size == (100, 100)
    # Cover scale (×5) fills the frame with red — no bars anywhere.
    assert cropped.getpixel((50, 0)) == (255, 0, 0)
    assert cropped.getpixel((50, 99)) == (255, 0, 0)

    padded = fit_image(src, 100, 100, "pad")
    assert padded.size == (100, 100)
    # Contain scale (×2.5) → 100×50 centered; top/bottom rows are neutral bars.
    assert padded.getpixel((50, 0)) == (0, 0, 0)
    assert padded.getpixel((50, 99)) == (0, 0, 0)
    assert padded.getpixel((50, 50)) == (255, 0, 0)

    # outpaint shares pad geometry (the bars become the generate region downstream).
    assert fit_image(src, 100, 100, "outpaint").getpixel((50, 0)) == (0, 0, 0)

    stretched = fit_image(src, 100, 100, "stretch")
    assert stretched.size == (100, 100)
    assert stretched.getpixel((50, 0)) == (255, 0, 0)  # distorted to fill, no bars

def test_load_source_image_honors_fit_mode_and_size_override(tmp_path, monkeypatch):
    # sc-2552: the source is fitted per request.fit_mode; explicit width/height override
    # the request dims (SenseNova snapped-bucket path).
    src = Image.new("RGB", (40, 20), (255, 0, 0))
    src_path = tmp_path / "src.png"
    src.save(src_path, "PNG")
    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: src_path,
    )
    # Output dims go through the 256-floor clamp in image_request_from_job, so use 400².
    pad_req = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "fitMode": "pad", "width": 400, "height": 400}}
    )
    padded = load_source_image(tmp_path, pad_req)
    assert padded.size == (400, 400)
    assert padded.getpixel((200, 0)) == (0, 0, 0)  # letterbox bar
    assert padded.getpixel((200, 200)) == (255, 0, 0)  # centered image

    # Default (crop) covers the frame — no bars.
    crop_req = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "width": 400, "height": 400}}
    )
    assert load_source_image(tmp_path, crop_req).getpixel((200, 0)) == (255, 0, 0)

    # Size override wins over the request dims (SenseNova snapped bucket); bypasses clamp.
    assert load_source_image(tmp_path, crop_req, 64, 48).size == (64, 48)

def test_load_mask_image_pad_keeps_bars_black(tmp_path, monkeypatch):
    # sc-2552: a padded mask must keep the bars black (= keep), only the original
    # region carries the user's white edit area, aligned with the padded source.
    mask = Image.new("RGB", (40, 20), "white")
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path, "PNG")
    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: mask_path,
    )
    request = image_request_from_job(
        {"payload": {"projectId": "p", "mode": "edit_image", "sourceAssetId": "asset_src",
                     "maskAssetId": "asset_mask", "fitMode": "pad", "width": 400, "height": 400}}
    )
    loaded = load_mask_image(tmp_path, request)
    assert loaded.mode == "L"
    assert loaded.size == (400, 400)
    assert loaded.getpixel((200, 0)) == 0  # padded bar = keep
    assert loaded.getpixel((200, 200)) == 255  # original region = edit

def test_outpaint_border_mask_marks_bars_white_keeps_center_black():
    # sc-2553: white = padded border to generate, black = centered source to keep,
    # geometry matching fit_image(..., "pad").
    src = Image.new("RGB", (40, 20), (255, 0, 0))  # 2:1 → top/bottom bars in a square box
    mask = outpaint_border_mask(src, 400, 400)
    assert mask.mode == "L"
    assert mask.size == (400, 400)
    assert mask.getpixel((200, 0)) == 255  # top bar = generate
    assert mask.getpixel((200, 399)) == 255  # bottom bar = generate
    assert mask.getpixel((200, 200)) == 0  # centered source = keep
    # A source already matching the box aspect leaves nothing to generate (all keep).
    flush = outpaint_border_mask(Image.new("RGB", (50, 50)), 400, 400)
    assert flush.getpixel((0, 0)) == 0 and flush.getpixel((200, 200)) == 0

def test_outpaint_border_mask_feather_softens_seam():
    # sc-2553: a feather introduces intermediate (anti-aliased) values at the seam;
    # the hard mask is strictly binary.
    src = Image.new("RGB", (40, 20), (255, 0, 0))
    hard = outpaint_border_mask(src, 400, 400, feather=0)
    soft = outpaint_border_mask(src, 400, 400, feather=20)
    column_hard = {hard.getpixel((200, y)) for y in range(400)}
    assert column_hard <= {0, 255}  # binary
    assert any(0 < soft.getpixel((200, y)) < 255 for y in range(400))  # gradient

def test_outpaint_feather_default_override_and_clamp():
    # sc-2553: feather defaults to ~1.5% of the short edge; advanced.outpaintFeather
    # overrides, clamped to [0, 128], bad values fall back to the default.
    base = {"projectId": "p", "mode": "edit_image", "sourceAssetId": "s",
            "width": 1024, "height": 768}
    default = round(768 * 0.015)
    assert SdxlDiffusersAdapter._outpaint_feather(image_request_from_job({"payload": base})) == default
    over = image_request_from_job({"payload": {**base, "advanced": {"outpaintFeather": 4}}})
    assert SdxlDiffusersAdapter._outpaint_feather(over) == 4
    huge = image_request_from_job({"payload": {**base, "advanced": {"outpaintFeather": 9999}}})
    assert SdxlDiffusersAdapter._outpaint_feather(huge) == 128
    bad = image_request_from_job({"payload": {**base, "advanced": {"outpaintFeather": "nope"}}})
    assert SdxlDiffusersAdapter._outpaint_feather(bad) == default

def test_run_image_upscale_requires_source_asset(monkeypatch, tmp_path):
    from scene_worker.image_adapters import run_image_upscale

    with pytest.raises(RuntimeError, match="source image asset"):
        run_image_upscale(
            SimpleNamespace(gpu_id=None),
            {"id": "job_x", "payload": {"projectId": "p"}},
            project_path=tmp_path,
            progress=lambda *args: None,
            cancel_requested=lambda: False,
        )

def test_model_supports_detail_matrix():
    # sc-2438: tile-ControlNet detail refine is SDXL-family only.
    assert model_supports_detail("sdxl") is True
    assert model_supports_detail("realvisxl") is True
    assert model_supports_detail("qwen_image_edit_2511") is False
    assert model_supports_detail("unknown_model") is False

def test_run_image_detail_writes_single_child_asset(tmp_path, monkeypatch):
    # sc-2438: standalone tile-ControlNet detail refine of an existing asset. Mock the
    # pipeline load + the tiled refine so the orchestration (source resolve → one child
    # asset + lineage + fact shape) runs without torch/diffusers/weights (CI has none).
    from scene_worker.image_adapters import run_image_detail

    source = Image.new("RGB", (12, 9), "green")
    source_path = tmp_path / "src.png"
    source.save(source_path, "PNG")
    refined = Image.new("RGB", (12, 9), "white")

    monkeypatch.setattr(
        "scene_worker.image_adapters.find_asset_media_path",
        lambda project_path, asset_id: source_path,
    )
    monkeypatch.setattr(
        "scene_worker.image_adapters._load_detail_pipeline",
        lambda settings, **kwargs: object(),
    )
    monkeypatch.setattr(
        "scene_worker.image_adapters._refine_tiled_detail",
        lambda pipe, image, **kwargs: (refined, 1),
    )

    job = {
        "id": "job_detail_1",
        "payload": {
            "projectId": "project_1",
            "sourceAssetId": "asset_src",
            "model": "realvisxl",
            "displayName": "Studio still",
            "advanced": {"strength": 0.6, "cnScale": 0.8},
        },
    }
    result = run_image_detail(
        SimpleNamespace(gpu_id=None),
        job,
        project_path=tmp_path,
        progress=lambda *args: None,
        cancel_requested=lambda: False,
    )

    writes = result["assetWrites"]
    assert len(writes) == 1
    fact = writes[0]
    # Detail refine preserves dimensions (refines in place — unlike upscale).
    assert (fact["width"], fact["height"]) == (12, 9)
    assert fact["mode"] == "image_detail"
    assert fact["model"] == "realvisxl"
    assert fact["sourceAssetId"] == "asset_src"
    assert fact["parents"] == ["asset_src"]
    assert fact["displayName"] == "Studio still (detail enhanced)"
    assert fact["extra"] == {
        "isDetailEnhanced": True,
        "detailFromAssetId": "asset_src",
        "backbone": "realvisxl",
        "strength": 0.6,
        "cnScale": 0.8,
    }
    written_path = tmp_path / fact["mediaPath"]
    assert written_path.is_file()
    assert Image.open(written_path).size == (12, 9)
    assert result["expectedCount"] == 1
    assert result["generationSet"]["id"] == result["generationSetId"]

def test_run_image_detail_requires_source_asset(tmp_path):
    from scene_worker.image_adapters import run_image_detail

    with pytest.raises(RuntimeError, match="source image asset"):
        run_image_detail(
            SimpleNamespace(gpu_id=None),
            {"id": "job_x", "payload": {"projectId": "p"}},
            project_path=tmp_path,
            progress=lambda *args: None,
            cancel_requested=lambda: False,
        )

def test_run_image_detail_rejects_non_detail_model(tmp_path):
    # A non-SDXL model is rejected before any pipeline load (no mocking needed).
    from scene_worker.image_adapters import run_image_detail

    with pytest.raises(RuntimeError, match="does not support detail"):
        run_image_detail(
            SimpleNamespace(gpu_id=None),
            {"id": "job_x", "payload": {"projectId": "p", "sourceAssetId": "asset_src",
                                        "model": "qwen_image_edit_2511"}},
            project_path=tmp_path,
            progress=lambda *args: None,
            cancel_requested=lambda: False,
        )

def test_gpu_worker_advertises_pose_detect_when_backend_installed(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "pose_detect" in capabilities

def test_gpu_worker_omits_pose_detect_without_backend(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.torch_inference_backend_available", lambda: True)
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: False)
    capabilities = worker_capabilities({"id": "gpu-0", "name": "GPU 0", "capabilities": ["placeholder", "gpu"]})
    assert "pose_detect" not in capabilities

def test_cpu_worker_never_advertises_pose_detect(monkeypatch):
    monkeypatch.setattr("scene_worker.runtime.pose_detector_backend_available", lambda: True)
    capabilities = worker_capabilities({"id": "cpu", "name": "CPU", "capabilities": ["placeholder", "cpu"]})
    assert capabilities == ["cpu"]

def test_loaded_models_are_collected_from_adapter_cache():
    class Adapter:
        def loaded_models(self):
            return ["Tongyi-MAI/Z-Image-Turbo"]

    assert loaded_models_from_adapters({"z": Adapter()}) == ["Tongyi-MAI/Z-Image-Turbo"]

def test_z_image_loaded_models_include_repo_and_model_id():
    adapter = ZImageDiffusersAdapter()
    adapter._loaded_repo = "Tongyi-MAI/Z-Image-Turbo"
    adapter._loaded_model = "z_image_turbo"

    assert set(adapter.loaded_models()) == {"Tongyi-MAI/Z-Image-Turbo", "z_image_turbo"}

def test_z_image_turbo_model_target_defaults_and_guidance_helper():
    target = MODEL_TARGETS["z_image_turbo"]
    assert target["adapter"] == "z_image_diffusers"
    assert target["family"] == "z-image"
    assert target["steps"] == 8
    assert target["guidanceScale"] == 1.0

    adapter = ZImageDiffusersAdapter()
    assert adapter._guidance_scale(SimpleNamespace(model="z_image_turbo", advanced={})) == 1.0
    assert adapter._guidance_scale(SimpleNamespace(model="z_image_turbo", advanced={"guidanceScale": 0.0})) == 0.0
    assert adapter._guidance_scale(SimpleNamespace(model="z_image_turbo", advanced={"guidanceScale": "bad"})) == 1.0

def test_qwen_loaded_models_track_text_and_edit_repos_independently():
    adapter = QwenImageAdapter()
    adapter._text_repo = "Qwen/Qwen-Image"
    adapter._edit_repo = "Qwen/Qwen-Image-Edit"
    adapter._loaded_model = "qwen_image_edit"

    assert set(adapter.loaded_models()) == {"Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit", "qwen_image_edit"}

def test_qwen_image_edit_2511_model_target_defaults():
    target = MODEL_TARGETS["qwen_image_edit_2511"]
    assert target["adapter"] == "qwen_image"
    assert target["family"] == "qwen-image"
    assert target["supportsEdit"] is True
    # Model card: 40 steps, guidanceScale 1.0, trueCfgScale 4.0 (default).
    assert target["steps"] == 40
    assert target["guidanceScale"] == 1.0
    assert target["repo"] == "Qwen/Qwen-Image-Edit-2511"

def test_qwen_image_edit_legacy_ids_alias_to_2511_repo():
    # sc-2160: the August qwen_image_edit and the September qwen_image_edit_2509
    # IDs now route to the December 2511 weights so old jobs/presets resolve.
    legacy = MODEL_TARGETS["qwen_image_edit"]
    sept = MODEL_TARGETS["qwen_image_edit_2509"]
    assert legacy["repo"] == "Qwen/Qwen-Image-Edit-2511"
    assert sept["repo"] == "Qwen/Qwen-Image-Edit-2511"
    assert legacy["steps"] == 40
    assert sept["steps"] == 40
    assert legacy["guidanceScale"] == 1.0
    assert sept["guidanceScale"] == 1.0

def test_qwen_image_edit_2511_lightning_model_target_defaults():
    target = MODEL_TARGETS["qwen_image_edit_2511_lightning"]
    assert target["adapter"] == "qwen_image"
    assert target["family"] == "qwen-image"
    assert target["supportsEdit"] is True
    # 4-step distill: cfg 1.0, true_cfg_scale 1.0, shares 2511 base weights.
    assert target["steps"] == 4
    assert target["guidanceScale"] == 1.0
    assert target["trueCfgScale"] == 1.0
    assert target["repo"] == "Qwen/Qwen-Image-Edit-2511"
    assert target["distillLora"] == {
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    }

def test_create_image_adapter_routes_qwen_image_edit_2511():
    # All Qwen Edit IDs ride the same QwenImageAdapter; pipeline-class selection
    # happens inside _load_pipeline based on the model id.
    adapter = create_image_adapter({"payload": {"model": "qwen_image_edit_2511"}})
    assert adapter.__class__.__name__ == "QwenImageAdapter"
    assert adapter.id == "qwen_image"
    lightning = create_image_adapter({"payload": {"model": "qwen_image_edit_2511_lightning"}})
    assert lightning.__class__.__name__ == "QwenImageAdapter"

def test_qwen_edit_pipeline_name_by_model():
    # sc-2160: every edit ID (incl. legacy aliases) ships the multi-image Plus
    # pipeline now that they all run against the 2511 base weights.
    assert QwenImageAdapter._edit_pipeline_name("qwen_image_edit") == "QwenImageEditPlusPipeline"
    assert QwenImageAdapter._edit_pipeline_name("qwen_image_edit_2509") == "QwenImageEditPlusPipeline"
    assert QwenImageAdapter._edit_pipeline_name("qwen_image_edit_2511") == "QwenImageEditPlusPipeline"
    assert QwenImageAdapter._edit_pipeline_name("qwen_image_edit_2511_lightning") == "QwenImageEditPlusPipeline"
    # Defensive: anything else still gets the single-image pipeline, matching
    # the same edit-style API surface.
    assert QwenImageAdapter._edit_pipeline_name("qwen_image") == "QwenImageEditPipeline"

def test_qwen_guidance_scale_reads_model_target_default():
    # 2511 family defaults to guidanceScale 1.0 (Plus pipeline requirement).
    request = SimpleNamespace(model="qwen_image_edit_2511", advanced={})
    assert QwenImageAdapter()._guidance_scale(request) == 1.0
    # qwen_image (text-to-image) still defaults to 4.0.
    request_t2i = SimpleNamespace(model="qwen_image", advanced={})
    assert QwenImageAdapter()._guidance_scale(request_t2i) == 4.0
    # User override wins.
    request_override = SimpleNamespace(model="qwen_image_edit_2511", advanced={"guidanceScale": 3.5})
    assert QwenImageAdapter()._guidance_scale(request_override) == 3.5

def test_qwen_true_cfg_scale_default_per_model():
    adapter = QwenImageAdapter()
    # Base family: 4.0 default.
    request = SimpleNamespace(model="qwen_image_edit_2511", advanced={})
    assert adapter._true_cfg_scale_default(request) == 4.0
    # Lightning: distilled at 1.0.
    lightning_request = SimpleNamespace(model="qwen_image_edit_2511_lightning", advanced={})
    assert adapter._true_cfg_scale_default(lightning_request) == 1.0
    # User override wins on either.
    override = SimpleNamespace(model="qwen_image_edit_2511_lightning", advanced={"trueCfgScale": 2.0})
    assert adapter._true_cfg_scale_default(override) == 2.0

def test_qwen_use_reference_only_for_character_image_with_reference():
    use = QwenImageAdapter._use_reference
    # Character Studio reference path requires both mode + a reference asset.
    assert use(SimpleNamespace(mode="character_image", reference_asset_id="a")) is True
    # No reference id → no character_image route (the picker shouldn't allow this
    # combination, but the worker double-checks).
    assert use(SimpleNamespace(mode="character_image", reference_asset_id=None)) is False
    # Edit and text-to-image modes go through their own paths regardless of
    # whether a reference id is also present.
    assert use(SimpleNamespace(mode="edit_image", reference_asset_id="a")) is False
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id="a")) is False

def test_qwen_reference_true_cfg_scale_default_and_clamp():
    adapter = QwenImageAdapter()
    # Model-card default 4.0; clamp [1, 10] (below 1 disables CFG and the edit
    # pipeline needs it > 1, above 10 collapses to pure negative-prompt steering).
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={})) == 4.0
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 2.5})) == 2.5
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 0.5})) == 1.0
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 99})) == 10.0
    assert adapter._reference_true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": "x"})) == 4.0

def test_qwen_reference_run_pipeline_passes_image_and_true_cfg(tmp_path, monkeypatch):
    """A character_image job with referenceAssetId drives the reference branch of
    _run_pipeline: load_reference_image(project_path, reference_asset_id) → image=
    kwarg, plus true_cfg_scale + negative_prompt (defaulted to ' ' so true CFG
    engages on Qwen edit pipelines). Mirrors the FLUX/SDXL torch-free pattern."""

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.last_kwargs: dict[str, Any] = {}

        # Named params so filter_call_kwargs keeps the Qwen-specific kwargs we
        # need to assert (FakePipe's `__call__` introspects clean via signature).
        def __call__(
            self,
            *,
            prompt=None,
            negative_prompt=None,
            image=None,
            true_cfg_scale=None,
            height=None,
            width=None,
            num_inference_steps=None,
            guidance_scale=None,
            generator=None,
            **kwargs,
        ):
            self.last_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "true_cfg_scale": true_cfg_scale,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                **kwargs,
            }
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen: list[tuple] = []

    def fake_load_reference_image(project_path, reference_asset_id):
        seen.append((project_path, reference_asset_id))
        return FakeImage()

    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image", fake_load_reference_image
    )

    # sampler_selection_from_advanced + apply_sampler poke the pipe — short-circuit.
    monkeypatch.setattr(
        "scene_worker.image_adapters.apply_sampler",
        lambda *args, **kwargs: None,
    )

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "character_image",
                "model": "qwen_image_edit_2509",
                "prompt": "the same character at a cafe",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {"trueCfgScale": 3.0},
            }
        }
    )
    pipe = FakePipe()
    result = QwenImageAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"), pipe, request, 7, project_path
    )
    # Reference branch ran: load_reference_image called, image= kwarg threaded
    # through with the loaded reference, true_cfg_scale from advanced overrides
    # default, and negative_prompt defaulted to " " so true CFG engages (Qwen's
    # edit pipeline needs a non-empty negative prompt for guidance > 1).
    assert seen == [(project_path, "asset-ref")]
    assert pipe.last_kwargs["true_cfg_scale"] == 3.0
    assert pipe.last_kwargs["negative_prompt"] == " "
    assert pipe.last_kwargs["image"] is not None
    assert not isinstance(pipe.last_kwargs["image"], list)  # single reference, no pose
    assert result is FakeOutput.images[0]

def test_qwen_reference_run_pipeline_threads_pose_skeleton_as_second_image(tmp_path, monkeypatch):
    """Best-effort pose tier (sc-2256): when _run_pipeline gets a pose_skeleton,
    the reference branch passes image=[reference, skeleton] to the multi-image
    edit pipeline (so the skeleton steers the body pose) and the prompt carries
    the pose cue. Without a skeleton it's a single-image reference (covered
    above)."""
    from scene_worker.character_studio_angles import POSE_SKELETON_PROMPT

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.last_kwargs: dict[str, Any] = {}

        def __call__(self, *, prompt=None, image=None, **kwargs):
            self.last_kwargs = {"prompt": prompt, "image": image, **kwargs}
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    reference_sentinel = FakeImage()
    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image",
        lambda project_path, reference_asset_id: reference_sentinel,
    )
    monkeypatch.setattr("scene_worker.image_adapters.apply_sampler", lambda *a, **k: None)

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "character_image",
                "model": "qwen_image_edit_2511_lightning",
                "prompt": "the same character",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {},
            }
        }
    )
    skeleton_sentinel = FakeImage()
    pipe = FakePipe()
    QwenImageAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"),
        pipe,
        request,
        7,
        project_path,
        prompt_override="the same character, " + POSE_SKELETON_PROMPT,
        pose_skeleton=skeleton_sentinel,
    )
    # Multi-image: reference first (identity), skeleton second (pose target).
    assert pipe.last_kwargs["image"] == [reference_sentinel, skeleton_sentinel]
    assert POSE_SKELETON_PROMPT in pipe.last_kwargs["prompt"]

def test_augment_prompt_for_pose_appends_skeleton_cue():
    from scene_worker.character_studio_angles import POSE_SKELETON_PROMPT, augment_prompt_for_pose

    assert augment_prompt_for_pose("a woman at a cafe") == f"a woman at a cafe, {POSE_SKELETON_PROMPT}"
    # Empty base → just the cue (caller still gets a usable instruction).
    assert augment_prompt_for_pose("") == POSE_SKELETON_PROMPT
    assert augment_prompt_for_pose("  a woman.  ") == f"a woman, {POSE_SKELETON_PROMPT}"

def test_qwen_strict_pose_entries_gating():
    """sc-2291: the base-Qwen strict pose path engages for a target that declares
    controlNetPose (base qwen_image) with advanced.poses — NOT for edit jobs, NOT for
    edit targets (no controlNetPose), and (unlike Kolors) with NO reference required."""
    from scene_worker.image_adapters import MODEL_TARGETS, QwenImageAdapter

    entries = QwenImageAdapter._strict_pose_entries
    base = MODEL_TARGETS["qwen_image"]  # declares controlNetPose
    edit = MODEL_TARGETS["qwen_image_edit_2511"]  # no controlNetPose → best-effort tier
    kp = [{"id": "sit", "keypoints": [[0.5, 0.5]] * 18}]
    # Base target + poses → fires, with OR without a reference (pose-from-prompt).
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={"poses": kp}), base) == kp
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id=None, advanced={"poses": kp}), base) == kp
    # Edit mode never takes the strict path.
    assert entries(SimpleNamespace(mode="edit_image", reference_asset_id="r", advanced={"poses": kp}), base) == []
    # Edit target (no controlNetPose) stays on the best-effort tier (sc-2256).
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={"poses": kp}), edit) == []
    # No poses → no strict path.
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={}), base) == []

def test_qwen_strict_pose_set_loops_poses_with_shared_seed(tmp_path, monkeypatch):
    """sc-2291: generate() on base qwen_image with advanced.poses routes to the strict
    ControlNet pose path — one image per pose, a single shared seed, hands/face threaded
    through, and poseLibrary + controlNetPose recorded in settings."""
    from scene_worker import image_adapters as ia

    monkeypatch.setattr(ia.QwenImageAdapter, "_load_pose_pipeline", lambda self, *a, **k: object())
    monkeypatch.setattr(ia.QwenImageAdapter, "_apply_loras", lambda self, *a, **k: None)
    monkeypatch.setattr(ia, "select_torch_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: SimpleNamespace() if name == "torch" else importlib.import_module(name),
    )

    calls: list[dict] = []

    def fake_run_pose(self, settings, pipe, request, seed, project_path, keypoints, hands, face, cancel_requested=None):
        from PIL import Image as _Image
        calls.append({"seed": seed, "keypoints": keypoints, "hands": hands, "face": face})
        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(ia.QwenImageAdapter, "_run_pose", fake_run_pose)

    captured: dict = {}

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, raw_settings, **kwargs):
            captured["raw_settings"] = raw_settings
            captured["image_count"] = image_count
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)

    kp = [[0.5, 0.1 + 0.04 * i] for i in range(18)]
    hands = [[[0.4, 0.4]] * 21, [[0.6, 0.4]] * 21]
    face = [[0.5, 0.3]] * 68
    job = {"id": "job_qwen_pose", "payload": {
        "projectId": "p", "mode": "character_image", "model": "qwen_image", "prompt": "the character",
        "count": 1, "width": 64, "height": 64,
        "advanced": {"poses": [
            {"id": "sit_01", "keypoints": kp},  # body-only
            {"id": "dance_01", "keypoints": kp, "hands": hands, "face": face},  # whole-body
        ]},
    }}
    ia.QwenImageAdapter().generate(
        settings=SimpleNamespace(gpu_id="cpu"), job=job, request=image_request_from_job(job),
        project_path=tmp_path, progress=lambda *a, **k: None, cancel_requested=lambda: False,
    )
    assert captured["image_count"] == 2  # one per pose, not request.count
    assert len(calls) == 2
    assert calls[0]["seed"] == calls[1]["seed"]  # shared seed across the set
    # Whole-body hands/face thread through to the DWPose-trained Qwen CN; body-only → None.
    assert calls[0]["hands"] is None and calls[0]["face"] is None
    assert calls[1]["hands"] is not None and len(calls[1]["face"]) == 68
    assert captured["raw_settings"].get("poseLibrary") is True
    assert captured["raw_settings"].get("controlNetPose") == "InstantX/Qwen-Image-ControlNet-Union"

def test_qwen_run_pose_conditions_controlnet_without_reference(tmp_path, monkeypatch):
    """sc-2291 strict tier: _run_pose feeds the rendered DWPose skeleton as control_image
    and controlScale as controlnet_conditioning_scale — pure txt2img + control, with NO
    reference / ip_adapter / img2img `image` kwarg (pose-from-prompt)."""
    import numpy as _np
    from scene_worker import image_adapters as ia

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.last_kwargs: dict = {}

        def __call__(self, *, prompt=None, control_image=None, controlnet_conditioning_scale=None, **kwargs):
            self.last_kwargs = {
                "prompt": prompt, "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale, **kwargs,
            }
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr("scene_worker.image_adapters.apply_sampler", lambda *a, **k: None)
    skeleton_sentinel = FakeImage()
    monkeypatch.setattr(
        ia, "draw_wholebody", lambda w, h, kps, hands=None, face=None, stickwidth=4: _np.zeros((h, w, 3), dtype=_np.uint8)
    )
    monkeypatch.setattr("scene_worker.image_adapters.Image", SimpleNamespace(fromarray=lambda arr: skeleton_sentinel))

    request = image_request_from_job({"payload": {
        "projectId": "p", "mode": "character_image", "model": "qwen_image",
        "prompt": "the character", "width": 16, "height": 16, "count": 1,
        "advanced": {"controlScale": 0.85},
    }})
    pipe = FakePipe()
    kp = [(0.5, 0.1 + 0.04 * i) for i in range(18)]
    ia.QwenImageAdapter()._run_pose(SimpleNamespace(gpu_id="cpu"), pipe, request, 7, tmp_path, kp, None, None)
    assert pipe.last_kwargs["control_image"] is skeleton_sentinel  # pose → ControlNet
    assert pipe.last_kwargs["controlnet_conditioning_scale"] == 0.85  # advanced.controlScale
    # Pose-from-prompt: no reference / IP-Adapter / img2img image kwarg.
    assert "image" not in pipe.last_kwargs
    assert "ip_adapter_image" not in pipe.last_kwargs

def test_image_adapter_env_aliases_and_unknown_values(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "procedural")
    assert create_image_adapter({"payload": {"model": "z_image_turbo"}}).__class__.__name__ == "ProceduralImageAdapter"

    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "typo")
    try:
        create_image_adapter({"payload": {"model": "z_image_turbo"}})
    except RuntimeError as exc:
        assert "Unsupported SCENEWORKS_IMAGE_ADAPTER" in str(exc)
    else:
        raise AssertionError("Unknown image adapter override should fail loudly.")

def test_create_image_adapter_lens_raises_candle_only():
    # sc-5126: Lens inference moved to the native candle (Windows/CUDA) backend and the Python
    # transformers-5 sidecar was retired. A lens job that reaches the Python worker (no candle worker
    # claimed it) must fail loudly rather than silently render the procedural placeholder — mirroring
    # the mlx_flux2 / pulid_flux arms.
    for model in ("lens", "lens_turbo"):
        try:
            create_image_adapter({"payload": {"model": model}})
        except RuntimeError as exc:
            assert "candle" in str(exc).lower()
        else:
            raise AssertionError(f"{model} must fail loudly on the Python worker after sc-5126.")

def test_image_adapter_env_override_lens_is_unsupported(monkeypatch):
    # `lens_turbo` is no longer a selectable Python image adapter (sc-5126), so an explicit override
    # is rejected like any unknown adapter id.
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "lens_turbo")
    try:
        create_image_adapter({"payload": {"model": "z_image_turbo"}})
    except RuntimeError as exc:
        assert "Unsupported SCENEWORKS_IMAGE_ADAPTER" in str(exc)
    else:
        raise AssertionError("lens_turbo override should be rejected after the sidecar retirement.")

def test_lens_turbo_model_target_defaults():
    target = MODEL_TARGETS["lens_turbo"]
    assert target["adapter"] == "lens_turbo"
    assert target["family"] == "lens"
    assert target["steps"] == 4
    assert target["supportsEdit"] is False
    assert target["repo"] == "microsoft/Lens-Turbo"

def test_lens_base_model_target_defaults():
    target = MODEL_TARGETS["lens"]
    assert target["adapter"] == "lens_turbo"
    assert target["family"] == "lens"
    # Non-distilled base: 20-step / CFG 5.0 (vs Turbo's 4 / 1.0).
    assert target["steps"] == 20
    assert target["guidanceScale"] == 5.0
    assert target["repo"] == "microsoft/Lens"

# Lens inference adapter tests (guidance defaults, edit rejection, sidecar-missing) were removed with
# the `LensTurboAdapter` class (sc-5126): the step/guidance defaults now live in the Rust MODEL_TABLE
# (`lens` 20/5.0, `lens_turbo` 4/1.0) and edit-shape rejection is enforced by the candle descriptor +
# the `image_request_candle_eligible` routing gate.

def test_create_image_adapter_routes_flux_schnell_and_dev():
    # Epic 3018 cutover (sc-3032): FLUX.1 on Mac is claimed by the Rust `mlx` GPU
    # worker; the Python worker always routes the torch FluxDiffusersAdapter.
    schnell = create_image_adapter({"payload": {"model": "flux_schnell"}})
    assert schnell.__class__.__name__ == "FluxDiffusersAdapter"
    assert schnell.id == "flux_diffusers"
    dev = create_image_adapter({"payload": {"model": "flux_dev"}})
    assert dev.__class__.__name__ == "FluxDiffusersAdapter"

def test_image_adapter_env_override_selects_flux(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "flux_diffusers")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "FluxDiffusersAdapter"

def test_flux_manifest_has_mlx_block():
    # Manifest-driven auto-dispatch + Model Manager memory tier (sc-1970).
    # The Rust API owns the canonical jsonc parser; here we just confirm both
    # FLUX entries carry an `mlx` block and the contents look right.
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()

    for model_id in ("flux_schnell", "flux_dev"):
        block = find_entry_block(model_id)
        mlx_block = find_mlx_block(block)
        quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
        mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
        assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
            f"{model_id} mlx.quantize must be a supported quant level (sc-1970)"
        )
        assert mem_match and int(mem_match.group(1)) > 0, (
            f"{model_id} mlx.minMemoryGb must be a positive int (sc-1970)"
        )

def test_qwen_image_manifest_has_mlx_block():
    # sc-1972: qwen_image carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2
    # precedent of restricting the menu to default-only when the MLX path is
    # the active backend, epic 1753 §14).
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("qwen_image")
    mlx_block = find_mlx_block(block)
    quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
        "qwen_image mlx.quantize must be a supported quant level (sc-1972)"
    )
    assert mem_match and int(mem_match.group(1)) > 0, (
        "qwen_image mlx.minMemoryGb must be a positive int (sc-1972)"
    )
    # MLX sampler/scheduler menu override
    assert '"samplers": ["default"]' in mlx_block, (
        "qwen_image mlx must restrict samplers to default (mflux loop is linear-only)"
    )
    assert '"schedulers": ["default"]' in mlx_block, (
        "qwen_image mlx must restrict schedulers to default (mflux loop is linear-only)"
    )

def test_create_image_adapter_routes_qwen_image():
    # Epic 3018 cutover (sc-3032): Qwen-Image on Mac is claimed by the Rust `mlx`
    # GPU worker; the Python worker always routes the torch QwenImageAdapter.
    adapter = create_image_adapter({"payload": {"model": "qwen_image"}})
    assert adapter.__class__.__name__ == "QwenImageAdapter"
    assert adapter.id == "qwen_image"

def test_flux2_true_v2_manifest_install_time_conversion():
    # sc-2235: the entry must declare the install-time conversion contract the
    # Rust convert job + adapter rely on.
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("flux2_klein_9b_true_v2")
    assert '"macOnly": true' in block
    assert '"adapter": "mlx_flux2"' in block
    # Only the bf16 single-file is pulled (not the whole 73 GB repo).
    assert "Flux2-Klein-9B-True-v2-bf16.safetensors" in block
    # Undistilled defaults differ from the 4-step distill.
    assert re.search(r'"steps"\s*:\s*24', block)

    mlx_block = find_mlx_block(block)
    assert '"requiresConversion": true' in mlx_block
    assert '"converter": "flux2_klein_diffusers"' in mlx_block
    assert '"convertSourceRepo": "wikeeyang/Flux2-Klein-9B-True-V2"' in mlx_block
    assert '"convertBaseRepo": "black-forest-labs/FLUX.2-klein-9B"' in mlx_block
    assert re.search(r'"quantize"\s*:\s*8', mlx_block)

def test_runtime_registry_covers_all_model_target_adapters():
    # sc-2203 guard: every adapter id a manifest model can dispatch to (the
    # `adapter` field in MODEL_TARGETS) must be a key in the runtime's
    # image_adapters registry, or create_image_adapter's `adapters.get(id)`
    # returns None and the job crashes. Catches "added to dispatch but forgot
    # to register" for any future adapter.
    #
    # Epic 3018 cutover (sc-3032): the MLX adapter ids are the exception — the
    # Python MLX adapters were deleted, so MLX-eligible models are claimed by the
    # Rust `mlx` GPU worker and never dispatch to a Python adapter (FLUX.2-klein
    # and PuLID-FLUX, the MLX-only families, make create_image_adapter raise). They
    # must NOT be registered on the Python worker.
    import re
    from pathlib import Path

    from scene_worker import runtime
    from scene_worker.image_adapters import MODEL_TARGETS

    # Adapter ids the Python worker no longer owns — claimed by a Rust NATIVE worker (MLX on Mac
    # and/or candle on Windows/CUDA), so create_image_adapter raises on them rather than dispatching
    # to a Python adapter. `pulid_flux` joined post-sc-3344 (torch PuLIDFluxAdapter retired);
    # `lens_turbo` joined post-sc-5126 (Lens inference sidecar retired — MLX on Mac, candle off-Mac).
    # Their MODEL_TARGETS rows keep the `adapter` id as the sentinel create_image_adapter raises on.
    rust_native_only = {"mlx_flux", "mlx_qwen", "mlx_z_image", "mlx_flux2", "pulid_flux", "lens_turbo"}

    src = Path(runtime.__file__).read_text(encoding="utf-8")
    block = src.split("image_adapters: dict[str, object] = {", 1)[1].split("}", 1)[0]
    registered = set(re.findall(r'"([a-z0-9_]+)":', block))

    needed = {
        target["adapter"]
        for target in MODEL_TARGETS.values()
        if target.get("adapter") and target["adapter"] not in rust_native_only
    }
    missing = needed - registered
    assert not missing, f"adapter ids in MODEL_TARGETS not registered in runtime: {sorted(missing)}"
    # The native-worker adapters are intentionally absent from the Python registry post-cutover.
    assert not (registered & rust_native_only)

def test_flux2_klein_manifest_entries_present():
    # Both flux2_klein_9b and flux2_klein_9b_kv must be present in the
    # builtin manifest with the expected adapter + family + mlx block.
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    # Both ids expose the same capability set: -kv is no longer gated to
    # character_image only — it runs plain txt2img on par with the base 9B,
    # the cache just doesn't engage without a reference (sc-2173).
    for model_id in ("flux2_klein_9b", "flux2_klein_9b_kv"):
        block = find_entry_block(model_id)
        assert '"adapter": "mlx_flux2"' in block, model_id
        assert '"family": "flux2-klein"' in block, model_id
        assert '"macOnly": true' in block, model_id
        assert '"gated": true' in block, model_id
        mlx_block = find_mlx_block(block)
        quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
        assert quant_match is not None, f"{model_id}: mlx.quantize missing"
        assert int(quant_match.group(1)) == 8, f"{model_id}: quantize should be 8 (sweet spot)"
        assert '"text_to_image"' in block, model_id
        assert '"character_image"' in block, model_id

def test_z_image_turbo_manifest_has_mlx_block():
    # sc-2145: z_image_turbo carries an mlx block + sampler/scheduler limits
    # override (mflux's loop is sealed on "linear" — match the wan_2_2 /
    # qwen_image precedents of restricting the menu to default-only when the
    # MLX path is the active backend, epic 1753 §14).
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("z_image_turbo")
    mlx_block = find_mlx_block(block)
    quant_match = re.search(r'"quantize"\s*:\s*(\d+)', mlx_block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert quant_match and int(quant_match.group(1)) in {3, 4, 5, 6, 8}, (
        "z_image_turbo mlx.quantize must be a supported quant level (sc-2145)"
    )
    assert mem_match and int(mem_match.group(1)) > 0, (
        "z_image_turbo mlx.minMemoryGb must be a positive int (sc-2145)"
    )
    assert '"samplers": ["default"]' in mlx_block, (
        "z_image_turbo mlx must restrict samplers to default (mflux loop is linear-only)"
    )
    assert '"schedulers": ["default"]' in mlx_block, (
        "z_image_turbo mlx must restrict schedulers to default (mflux loop is linear-only)"
    )

def test_create_image_adapter_routes_z_image_turbo():
    # Epic 3018 cutover (sc-3032): Z-Image on Mac is claimed by the Rust `mlx` GPU
    # worker; the Python worker always routes the torch ZImageDiffusersAdapter.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "ZImageDiffusersAdapter"
    assert adapter.id == "z_image_diffusers"

def test_sdxl_manifest_has_mlx_block():
    # sc-1975: sdxl carries an mlx block (no `limits` override here — Apple's
    # SDXL schedule already matches the torch EulerDiscrete default, and
    # there's no per-model sampler menu in the sdxl manifest entry to limit).
    import re

    _, find_entry_block, find_mlx_block = _manifest_brace_walker()
    block = find_entry_block("sdxl")
    mlx_block = find_mlx_block(block)
    mem_match = re.search(r'"minMemoryGb"\s*:\s*(\d+)', mlx_block)
    assert mem_match and int(mem_match.group(1)) > 0, (
        "sdxl mlx.minMemoryGb must be a positive int (sc-1975)"
    )
    # No quantize key in v1 — Apple's Q8 recipe breaks SDXL base 1.0 (see
    # sc-1975 spike finding). bf16 is the only supported precision.
    assert '"quantize"' not in mlx_block, (
        "sdxl mlx block must not declare a quantize default in v1 (sc-1975: "
        "Apple's Q8 recipe breaks on SDXL base 1.0; defer until calibration lands)"
    )

def test_sdxl_auto_dispatch_uses_torch_adapter_on_python_worker(monkeypatch):
    # sc-3060 retired the in-process vendored MLX SDXL adapter (sc-1975). On the Python
    # worker, SDXL now always resolves to the torch SdxlDiffusersAdapter — on every host
    # and for every shape (txt2img / edit / reference). MLX routing for SDXL lives in the
    # Rust API claim layer (jobs_store::sdxl_mlx_eligible), which defers eligible jobs to
    # the `mlx` GPU worker; those never reach this Python adapter.
    monkeypatch.delenv("SCENEWORKS_IMAGE_ADAPTER", raising=False)
    for platform in ("darwin", "linux", "win32"):
        monkeypatch.setattr("sys.platform", platform)
        for payload in (
            {"model": "sdxl"},
            {"model": "sdxl", "mode": "edit_image"},
            {"model": "sdxl", "referenceAssetId": "asset_ref"},
            {"model": "realvisxl"},
        ):
            adapter = create_image_adapter({"payload": payload})
            assert adapter.__class__.__name__ == "SdxlDiffusersAdapter", (
                f"{payload} on {platform} must use the torch SDXL adapter"
            )

def test_flux_model_target_defaults():
    schnell = MODEL_TARGETS["flux_schnell"]
    assert schnell["adapter"] == "flux_diffusers"
    assert schnell["family"] == "flux"
    assert schnell["supportsEdit"] is False
    # Guidance-distilled: 4 steps, guidance 0, T5 max_seq_len 256.
    assert schnell["steps"] == 4
    assert schnell["guidanceScale"] == 0.0
    assert schnell["maxSequenceLength"] == 256
    assert schnell["repo"] == "black-forest-labs/FLUX.1-schnell"

    dev = MODEL_TARGETS["flux_dev"]
    assert dev["adapter"] == "flux_diffusers"
    assert dev["family"] == "flux"
    assert dev["supportsEdit"] is False
    # Guided: 28 steps, guidance 3.5, T5 max_seq_len 512.
    assert dev["steps"] == 28
    assert dev["guidanceScale"] == 3.5
    assert dev["maxSequenceLength"] == 512
    assert dev["repo"] == "black-forest-labs/FLUX.1-dev"
    # XLabs FLUX IP-Adapter is the diffusers-blessed character-image path; CLIP-L
    # encoder, no subfolder. flux_schnell has no native IP-Adapter trained for it,
    # so the block lives on flux_dev only (sc-2011).
    ip_adapter = dev["ipAdapter"]
    assert ip_adapter["repo"] == "XLabs-AI/flux-ip-adapter"
    assert ip_adapter["weight"] == "ip_adapter.safetensors"
    assert ip_adapter["imageEncoderRepo"] == "openai/clip-vit-large-patch14"
    assert "ipAdapter" not in schnell

def test_flux_guidance_scale_uses_per_model_default_and_override():
    adapter = FluxDiffusersAdapter()
    schnell = MODEL_TARGETS["flux_schnell"]
    dev = MODEL_TARGETS["flux_dev"]
    # Per-model default applies when the request does not override guidance.
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), schnell) == 0.0
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), dev) == 3.5
    # An explicit request value wins.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 2.0}), dev) == 2.0
    # Unparseable override falls back to the per-model default.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": "x"}), dev) == 3.5

def test_flux_num_inference_steps_default_and_override():
    adapter = FluxDiffusersAdapter()
    schnell = MODEL_TARGETS["flux_schnell"]
    dev = MODEL_TARGETS["flux_dev"]
    # Defaults come straight from the model target (no Z-Image +1 quirk).
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), schnell) == 4
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), dev) == 28
    # Explicit override is honored and clamped to [1, 80].
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 12}), schnell) == 12
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), dev) == 80

def test_flux_max_sequence_length_default_and_override():
    adapter = FluxDiffusersAdapter()
    schnell = MODEL_TARGETS["flux_schnell"]
    dev = MODEL_TARGETS["flux_dev"]
    assert adapter._max_sequence_length(SimpleNamespace(advanced={}), schnell) == 256
    assert adapter._max_sequence_length(SimpleNamespace(advanced={}), dev) == 512
    # Override honored, clamped to the T5 max of 512.
    assert adapter._max_sequence_length(SimpleNamespace(advanced={"maxSequenceLength": 128}), dev) == 128
    assert adapter._max_sequence_length(SimpleNamespace(advanced={"maxSequenceLength": 4096}), dev) == 512

def test_flux_reference_asset_id_parsed():
    request = image_request_from_job(
        {"payload": {"projectId": "p", "model": "flux_dev", "referenceAssetId": "asset-ref"}}
    )
    assert request.reference_asset_id == "asset-ref"

def test_flux_use_ip_adapter_only_for_text_with_reference():
    use = FluxDiffusersAdapter._use_ip_adapter
    # IP-Adapter runs on the T2I pipeline with a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id="a")) is True
    # FLUX is T2I-only today but the gate still mirrors SDXL/Kolors so reference +
    # img2img (FLUX.1 Kontext) can opt in cleanly when that path lands.
    assert use(SimpleNamespace(mode="edit_image", reference_asset_id="a")) is False
    # No reference image → no IP-Adapter regardless of mode.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id=None)) is False

def test_flux_ip_adapter_scale_default_and_clamp():
    adapter = FluxDiffusersAdapter()
    # XLabs+CLIP-L is the resemblance tier (faithful identity = PuLID-FLUX); the
    # default 0.7 matches SDXL plus-face — same headroom for the prompt.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={})) == 0.7
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 0.4})) == 0.4
    # Clamped to [0, 1]; unparseable falls back to the default.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 5})) == 1.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": -2})) == 0.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": "x"})) == 0.7

def test_flux_true_cfg_scale_default_and_clamp():
    adapter = FluxDiffusersAdapter()
    # FLUX is guidance-distilled, so real CFG against negative_prompt rides on
    # the parallel true_cfg_scale kwarg. XLabs docs default 4.0; clamp [1, 10]
    # (below 1.0 disables CFG, above 10 hard-bakes the negative prompt).
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={})) == 4.0
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 2.5})) == 2.5
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 0.0})) == 1.0
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": 99})) == 10.0
    assert adapter._true_cfg_scale(SimpleNamespace(advanced={"trueCfgScale": "x"})) == 4.0

def test_flux_reference_run_pipeline_passes_ip_adapter_image_and_true_cfg(tmp_path, monkeypatch):
    """A T2I FLUX job with referenceAssetId drives the IP-Adapter branch of
    _run_pipeline: load_reference_image(project_path, reference_asset_id) →
    ip_adapter_image kwarg, set_ip_adapter_scale() per request, and true_cfg_scale
    + negative_prompt onto kwargs. Mirrors the SDXL torch-free pattern."""

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.scales: list[float] = []
            self.last_kwargs: dict[str, Any] = {}

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        # Named params so filter_call_kwargs keeps the FLUX-specific kwargs we
        # need to assert (it introspects __call__ via inspect.signature and drops
        # anything not in the accepted-name set; a bare **kwargs is treated as
        # accepting nothing because the var-keyword param itself is the only
        # name in the signature).
        def __call__(
            self,
            *,
            prompt=None,
            negative_prompt=None,
            ip_adapter_image=None,
            true_cfg_scale=None,
            height=None,
            width=None,
            num_inference_steps=None,
            guidance_scale=None,
            max_sequence_length=None,
            generator=None,
            **kwargs,
        ):
            self.last_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "ip_adapter_image": ip_adapter_image,
                "true_cfg_scale": true_cfg_scale,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_sequence_length,
                "generator": generator,
                **kwargs,
            }
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen: list[tuple] = []

    def fake_load_reference_image(project_path, reference_asset_id):
        seen.append((project_path, reference_asset_id))
        return FakeImage()

    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image", fake_load_reference_image
    )

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_image",
                "model": "flux_dev",
                "prompt": "a portrait of the character",
                "negativePrompt": "blurry",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {"ipAdapterScale": 0.5, "trueCfgScale": 3.0},
            }
        }
    )
    pipe = FakePipe()
    result = FluxDiffusersAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"), pipe, request, 7, project_path
    )
    # IP-Adapter branch ran: reference loaded → ip_adapter_image kwarg, per-request
    # scale applied, and true_cfg_scale + negative_prompt threaded through.
    assert seen == [(project_path, "asset-ref")]
    assert pipe.scales == [0.5]
    assert pipe.last_kwargs["true_cfg_scale"] == 3.0
    assert pipe.last_kwargs["negative_prompt"] == "blurry"
    assert pipe.last_kwargs["ip_adapter_image"] is not None
    assert result is FakeOutput.images[0]

def test_flux_rejects_image_edit():
    job = {
        "id": "job_flux_edit",
        "payload": {
            "projectId": "project_x",
            "mode": "edit_image",
            "model": "flux_dev",
            "prompt": "a cat",
        },
    }
    noop = lambda *args, **kwargs: None  # noqa: E731
    try:
        FluxDiffusersAdapter().generate(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )
    except RuntimeError as exc:
        assert "does not support image editing" in str(exc)
    else:
        raise AssertionError("FLUX.1 is text-to-image only and must reject edit_image.")

def test_create_image_adapter_routes_chroma_variants():
    for model in ("chroma1_hd", "chroma1_base", "chroma1_flash"):
        adapter = create_image_adapter({"payload": {"model": model}})
        assert adapter.__class__.__name__ == "ChromaDiffusersAdapter"
        assert adapter.id == "chroma_diffusers"

def test_image_adapter_env_override_selects_chroma(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "chroma_diffusers")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "ChromaDiffusersAdapter"

def test_chroma_model_target_defaults():
    hd = MODEL_TARGETS["chroma1_hd"]
    base = MODEL_TARGETS["chroma1_base"]
    flash = MODEL_TARGETS["chroma1_flash"]
    for target in (hd, base, flash):
        assert target["adapter"] == "chroma_diffusers"
        assert target["family"] == "chroma"
        assert target["supportsEdit"] is False
        assert target["maxSequenceLength"] == 512
    # HD/Base: real CFG with negative prompts (~40 steps, guidance 3.0).
    assert hd["steps"] == 40
    assert hd["guidanceScale"] == 3.0
    assert hd["repo"] == "lodestones/Chroma1-HD"
    assert base["steps"] == 40
    assert base["guidanceScale"] == 3.0
    assert base["repo"] == "lodestones/Chroma1-Base"
    # Flash: CFG baked off (~8 steps, guidance 1.0).
    assert flash["steps"] == 8
    assert flash["guidanceScale"] == 1.0
    assert flash["repo"] == "lodestones/Chroma1-Flash"

def test_chroma_guidance_scale_uses_per_model_default_and_override():
    adapter = ChromaDiffusersAdapter()
    hd = MODEL_TARGETS["chroma1_hd"]
    flash = MODEL_TARGETS["chroma1_flash"]
    # Per-model default applies when the request does not override guidance.
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), hd) == 3.0
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), flash) == 1.0
    # An explicit request value wins.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 4.5}), hd) == 4.5
    # Unparseable override falls back to the per-model default.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": "x"}), hd) == 3.0

def test_chroma_num_inference_steps_default_and_override():
    adapter = ChromaDiffusersAdapter()
    hd = MODEL_TARGETS["chroma1_hd"]
    flash = MODEL_TARGETS["chroma1_flash"]
    # Defaults come straight from the model target.
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), hd) == 40
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), flash) == 8
    # Explicit override is honored and clamped to [1, 80].
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 20}), hd) == 20
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), flash) == 80

def test_chroma_rejects_image_edit():
    job = {
        "id": "job_chroma_edit",
        "payload": {
            "projectId": "project_x",
            "mode": "edit_image",
            "model": "chroma1_hd",
            "prompt": "a cat",
        },
    }
    noop = lambda *args, **kwargs: None  # noqa: E731
    try:
        ChromaDiffusersAdapter().generate(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )
    except RuntimeError as exc:
        assert "does not support image editing" in str(exc)
    else:
        raise AssertionError("Chroma1 is text-to-image only and must reject edit_image.")

def test_create_image_adapter_routes_kolors():
    adapter = create_image_adapter({"payload": {"model": "kolors"}})
    assert adapter.__class__.__name__ == "KolorsDiffusersAdapter"
    assert adapter.id == "kolors_diffusers"

def test_image_adapter_env_override_selects_kolors(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "kolors_diffusers")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "KolorsDiffusersAdapter"

def test_kolors_model_target_defaults():
    kolors = MODEL_TARGETS["kolors"]
    assert kolors["adapter"] == "kolors_diffusers"
    assert kolors["family"] == "kolors"
    # Unified checkpoint does both T2I (KolorsPipeline) and img2img edit.
    assert kolors["supportsEdit"] is True
    # Real CFG (not distilled): guidance 5.0, ~25 steps, ChatGLM3 max_seq_len 256.
    assert kolors["steps"] == 25
    assert kolors["guidanceScale"] == 5.0
    assert kolors["maxSequenceLength"] == 256
    # fp16-only repo: from_pretrained must request the fp16 variant.
    assert kolors["variant"] == "fp16"
    assert kolors["repo"] == "Kwai-Kolors/Kolors-diffusers"

def test_kolors_guidance_scale_uses_per_model_default_and_override():
    adapter = KolorsDiffusersAdapter()
    kolors = MODEL_TARGETS["kolors"]
    # Unlike the guidance-distilled FLUX path, Kolors defaults to real CFG (5.0).
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), kolors) == 5.0
    # An explicit request value wins.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 7.0}), kolors) == 7.0
    # Unparseable override falls back to the per-model default.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": "x"}), kolors) == 5.0

def test_kolors_num_inference_steps_default_and_override():
    adapter = KolorsDiffusersAdapter()
    kolors = MODEL_TARGETS["kolors"]
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), kolors) == 25
    # Explicit override is honored and clamped to [1, 80].
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 30}), kolors) == 30
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), kolors) == 80

def test_kolors_max_sequence_length_default_and_override():
    adapter = KolorsDiffusersAdapter()
    kolors = MODEL_TARGETS["kolors"]
    assert adapter._max_sequence_length(SimpleNamespace(advanced={}), kolors) == 256
    # Override honored, clamped to the ChatGLM3 max of 256.
    assert adapter._max_sequence_length(SimpleNamespace(advanced={"maxSequenceLength": 128}), kolors) == 128
    assert adapter._max_sequence_length(SimpleNamespace(advanced={"maxSequenceLength": 4096}), kolors) == 256

def test_kolors_supports_edit():
    # Kolors is a unified checkpoint: KolorsPipeline (T2I) + KolorsImg2ImgPipeline (edit).
    assert model_supports_edit("kolors") is True

def test_create_image_adapter_routes_kolors_edit():
    # Edit jobs route to the same adapter; the adapter switches pipeline by mode.
    adapter = create_image_adapter({"payload": {"model": "kolors", "mode": "edit_image"}})
    assert adapter.__class__.__name__ == "KolorsDiffusersAdapter"
    assert adapter.id == "kolors_diffusers"

def test_kolors_reference_asset_id_parsed():
    request = image_request_from_job(
        {"payload": {"projectId": "p", "model": "kolors", "referenceAssetId": "asset-ref"}}
    )
    assert request.reference_asset_id == "asset-ref"

def test_kolors_use_ip_adapter_only_for_text_with_reference():
    use = KolorsDiffusersAdapter._use_ip_adapter
    # IP-Adapter runs on the T2I pipeline with a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id="a")) is True
    # Not for edit jobs (reference + img2img together is a future enhancement)...
    assert use(SimpleNamespace(mode="edit_image", reference_asset_id="a")) is False
    # ...and not without a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id=None)) is False

def test_kolors_ip_adapter_scale_default_and_clamp():
    adapter = KolorsDiffusersAdapter()
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={})) == 0.6
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 0.4})) == 0.4
    # Clamped to [0, 1]; unparseable falls back to the default.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 5})) == 1.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": -2})) == 0.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": "x"})) == 0.6

def test_load_reference_image_requires_asset(tmp_path):
    try:
        load_reference_image(tmp_path, "")
    except RuntimeError as exc:
        assert "reference image asset" in str(exc)
    else:
        raise AssertionError("load_reference_image must reject a missing reference asset id.")

def test_kolors_reference_run_pipeline_passes_ip_adapter_image(tmp_path, monkeypatch):
    """A T2I job with referenceAssetId drives the IP-Adapter branch of _run_pipeline:
    load_reference_image(project_path, reference_asset_id) → ip_adapter_image kwarg, plus
    a per-request set_ip_adapter_scale. Mirrors test_edit_run_pipeline_threads_project_path
    so it runs torch-free in CI."""

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.scales: list[float] = []

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen: list[tuple] = []

    def fake_load_reference_image(project_path, reference_asset_id):
        seen.append((project_path, reference_asset_id))
        return FakeImage()

    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image", fake_load_reference_image
    )

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_image",
                "model": "kolors",
                "prompt": "a portrait of the character",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {"ipAdapterScale": 0.7},
            }
        }
    )
    pipe = FakePipe()
    result = KolorsDiffusersAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"), pipe, request, 7, project_path
    )
    # The IP-Adapter branch ran: it loaded the reference image (→ ip_adapter_image
    # kwarg) and applied the per-request scale. (filter_call_kwargs only keeps a
    # pipe's *named* params, and FakePipe takes **kwargs, so we assert via the
    # observable side effects rather than last_kwargs — same as the edit test.)
    assert seen == [(project_path, "asset-ref")]
    assert pipe.scales == [0.7]
    assert result is FakeOutput.images[0]

def test_kolors_pose_entries_gating():
    """sc-2264: the pose ControlNet path engages only for a character_image job WITH a
    reference and advanced.poses — not for edit_image, and not without a reference."""
    entries = KolorsDiffusersAdapter._pose_entries
    kp = [{"id": "sit", "keypoints": [[0.5, 0.5]] * 18}]
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={"poses": kp})) == kp
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id=None, advanced={"poses": kp})) == []
    assert entries(SimpleNamespace(mode="edit_image", reference_asset_id="r", advanced={"poses": kp})) == []
    assert entries(SimpleNamespace(mode="character_image", reference_asset_id="r", advanced={})) == []

def test_kolors_run_pose_composes_skeleton_controlnet_and_ip_adapter(tmp_path, monkeypatch):
    """sc-2264 strict tier: _run_pose feeds the rendered skeleton as control_image, the
    reference as both ip_adapter_image (identity) and img2img init, and the openPoseScale
    as controlnet_conditioning_scale — pose ControlNet + IP-Adapter in one call."""
    import numpy as _np
    from scene_worker import image_adapters as ia

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.last_kwargs: dict = {}
            self.scales: list = []

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        def __call__(self, *, prompt=None, image=None, control_image=None, ip_adapter_image=None,
                     controlnet_conditioning_scale=None, **kwargs):
            self.last_kwargs = {
                "image": image, "control_image": control_image, "ip_adapter_image": ip_adapter_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale, **kwargs,
            }
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    skeleton_sentinel = FakeImage()
    reference_sentinel = FakeImage()
    # Kolors _run_pose renders the DWPose whole-body skeleton (sc-2289); draw_wholebody
    # needs cv2 (absent in the CI venv), so stub it (real render covered elsewhere).
    monkeypatch.setattr(
        ia,
        "draw_wholebody",
        lambda w, h, kps, hands=None, face=None, stickwidth=4: _np.zeros((h, w, 3), dtype=_np.uint8),
    )
    monkeypatch.setattr("scene_worker.image_adapters.Image", SimpleNamespace(fromarray=lambda arr: skeleton_sentinel))
    monkeypatch.setattr(ia, "load_reference_image", lambda project_path, asset_id: reference_sentinel)

    request = image_request_from_job({"payload": {
        "projectId": "p", "mode": "character_image", "model": "kolors",
        "prompt": "the character", "referenceAssetId": "asset-ref",
        "width": 16, "height": 16, "count": 1, "advanced": {"openPoseScale": 0.65, "ipAdapterScale": 0.6},
    }})
    pipe = FakePipe()
    kp = [(0.5, 0.1 + 0.04 * i) for i in range(18)]
    KolorsDiffusersAdapter()._run_pose(SimpleNamespace(gpu_id="cpu"), pipe, request, 7, tmp_path, kp)
    assert pipe.last_kwargs["control_image"] is skeleton_sentinel  # pose -> ControlNet
    assert pipe.last_kwargs["ip_adapter_image"] is reference_sentinel  # identity -> IP-Adapter
    assert pipe.last_kwargs["image"] is reference_sentinel  # img2img init = reference
    assert pipe.last_kwargs["controlnet_conditioning_scale"] == 0.65
    assert pipe.scales == [0.6]

def test_kolors_pose_set_loops_poses_with_shared_seed(tmp_path, monkeypatch):
    """sc-2264: generate() with advanced.poses routes to the ControlNet pose path — one
    image per library pose, a single shared seed, and poseLibrary recorded in settings."""
    from scene_worker import image_adapters as ia

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_load_pose_pipeline", lambda self, *a, **k: object())
    monkeypatch.setattr(ia, "select_torch_device", lambda *a, **k: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: SimpleNamespace() if name == "torch" else importlib.import_module(name),
    )

    calls: list[dict] = []

    def fake_run_pose(self, settings, pipe, request, seed, project_path, keypoints, hands=None, face=None, cancel_requested=None):
        from PIL import Image as _Image
        calls.append({"seed": seed, "keypoints": keypoints, "hands": hands, "face": face})
        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(ia.KolorsDiffusersAdapter, "_run_pose", fake_run_pose)

    captured: dict = {}

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, raw_settings, **kwargs):
            captured["raw_settings"] = raw_settings
            captured["image_count"] = image_count
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)

    kp = [[0.5, 0.1 + 0.04 * i] for i in range(18)]
    hands = [[[0.4, 0.4]] * 21, [[0.6, 0.4]] * 21]
    face = [[0.5, 0.3]] * 68
    job = {"id": "job_kolors_pose", "payload": {
        "projectId": "p", "mode": "character_image", "model": "kolors", "prompt": "the character",
        "referenceAssetId": "ref-1", "count": 1, "width": 64, "height": 64,
        "advanced": {"poses": [
            {"id": "sit_01", "keypoints": kp},  # body-only
            {"id": "dance_01", "keypoints": kp, "hands": hands, "face": face},  # whole-body
        ]},
    }}
    KolorsDiffusersAdapter().generate(
        settings=SimpleNamespace(gpu_id="cpu"), job=job, request=image_request_from_job(job),
        project_path=tmp_path, progress=lambda *a, **k: None, cancel_requested=lambda: False,
    )
    assert captured["image_count"] == 2  # one per pose, not request.count
    assert len(calls) == 2
    assert calls[0]["seed"] == calls[1]["seed"]  # shared seed across the set
    # sc-2289: whole-body hands/face thread through to the DWPose-trained Kolors CN;
    # body-only poses pass None (rendered identically to the old body-only path).
    assert calls[0]["hands"] is None and calls[0]["face"] is None
    assert calls[1]["hands"] is not None and len(calls[1]["face"]) == 68
    assert captured["raw_settings"].get("poseLibrary") is True
    assert captured["raw_settings"].get("controlNetPose") == "Kwai-Kolors/Kolors-ControlNet-Pose"

def test_create_image_adapter_routes_sdxl(monkeypatch):
    # SDXL resolves to the torch adapter on the Python worker (sc-3060 retired the
    # vendored MLX SDXL path; Mac MLX routing now lives in the Rust API claim layer).
    monkeypatch.delenv("SCENEWORKS_IMAGE_ADAPTER", raising=False)
    adapter = create_image_adapter({"payload": {"model": "sdxl"}})
    assert adapter.__class__.__name__ == "SdxlDiffusersAdapter"
    assert adapter.id == "sdxl_diffusers"

def test_image_adapter_env_override_selects_sdxl(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "sdxl_diffusers")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "SdxlDiffusersAdapter"

def test_sdxl_model_target_defaults():
    sdxl = MODEL_TARGETS["sdxl"]
    assert sdxl["adapter"] == "sdxl_diffusers"
    assert sdxl["family"] == "sdxl"
    # Unified base checkpoint does both T2I and img2img edit.
    assert sdxl["supportsEdit"] is True
    # Real CFG with negative prompt: ~30 steps at guidance 7.0.
    assert sdxl["steps"] == 30
    assert sdxl["guidanceScale"] == 7.0
    # fp16 variant; two CLIP encoders so there is no max_sequence_length knob.
    assert sdxl["variant"] == "fp16"
    assert "maxSequenceLength" not in sdxl
    assert sdxl["repo"] == "stabilityai/stable-diffusion-xl-base-1.0"
    # IP-Adapter plus-face for Character Studio reference conditioning (sc-2007):
    # ViT-H image encoder shipped in the same repo, sdxl_models subfolder for the
    # weights. Plus-face is the identity-leaning variant without scene/outfit copy.
    ip_adapter = sdxl["ipAdapter"]
    assert ip_adapter["repo"] == "h94/IP-Adapter"
    assert ip_adapter["subfolder"] == "sdxl_models"
    assert ip_adapter["weight"] == "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    assert ip_adapter["encoderSubfolder"] == "models/image_encoder"

def test_sdxl_supports_edit():
    # SDXL base is a unified checkpoint: StableDiffusionXLPipeline (T2I) +
    # StableDiffusionXLImg2ImgPipeline (edit).
    assert model_supports_edit("sdxl") is True

def test_create_image_adapter_routes_sdxl_edit():
    # Edit jobs route to the same adapter; it switches pipeline by mode.
    adapter = create_image_adapter({"payload": {"model": "sdxl", "mode": "edit_image"}})
    assert adapter.__class__.__name__ == "SdxlDiffusersAdapter"
    assert adapter.id == "sdxl_diffusers"

def test_sdxl_guidance_scale_uses_per_model_default_and_override():
    adapter = SdxlDiffusersAdapter()
    sdxl = MODEL_TARGETS["sdxl"]
    # SDXL uses real CFG; the per-model default (7.0) applies without an override.
    assert adapter._guidance_scale(SimpleNamespace(advanced={}), sdxl) == 7.0
    # An explicit request value wins.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": 5.0}), sdxl) == 5.0
    # Unparseable override falls back to the per-model default.
    assert adapter._guidance_scale(SimpleNamespace(advanced={"guidanceScale": "x"}), sdxl) == 7.0

def test_zimage_num_inference_steps_matches_manifest_default():
    # sc-4188 drift fix: ZImage defaulted to model_target["steps"] + 1 (9) with
    # no rationale while every other adapter — and the MLX engine path — uses
    # the manifest value unmodified (8).
    from scene_worker.image_adapters import ZImageDiffusersAdapter

    adapter = ZImageDiffusersAdapter()
    target = MODEL_TARGETS["z_image_turbo"]
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), target) == target["steps"] == 8
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 12}), target) == 12

def test_generate_images_always_wires_upscaler_telemetry(monkeypatch):
    # sc-4188 drift fix: five adapters dropped settings=/job_id= from the
    # incremental writer call, silently disabling upscaler telemetry/job-id
    # wiring. The shared _generate_images passes them unconditionally.
    from scene_worker.image_adapters import FluxDiffusersAdapter, ImageAssetWriter

    captured: dict = {}

    def fake_writer(self, **kwargs):
        captured.update(kwargs)
        return {"images": []}

    monkeypatch.setattr(ImageAssetWriter, "write_incremental_outputs", fake_writer)
    fake_torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
                                 cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: fake_torch if name == "torch" else importlib.import_module(name),
    )
    adapter = FluxDiffusersAdapter()
    settings = SimpleNamespace(gpu_id="cpu")
    request = SimpleNamespace(model="flux_schnell", seed=1, seeds=[], prompt="p", advanced={})
    adapter._generate_images(
        settings=settings,
        job={"id": "job-telemetry"},
        request=request,
        project_path=Path("."),
        progress=lambda *a, **k: None,
        cancel_requested=lambda: False,
        label="FLUX.1",
        image_count=1,
        run_one=lambda index, seed: FakeImage(),
        raw_settings={"realModelInference": True},
    )
    assert captured["settings"] is settings
    assert captured["job_id"] == "job-telemetry"

def test_sdxl_num_inference_steps_default_and_override():
    adapter = SdxlDiffusersAdapter()
    sdxl = MODEL_TARGETS["sdxl"]
    assert adapter._num_inference_steps(SimpleNamespace(advanced={}), sdxl) == 30
    # Explicit override is honored and clamped to [1, 80].
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 45}), sdxl) == 45
    assert adapter._num_inference_steps(SimpleNamespace(advanced={"steps": 999}), sdxl) == 80

def test_sdxl_reference_asset_id_parsed():
    request = image_request_from_job(
        {"payload": {"projectId": "p", "model": "sdxl", "referenceAssetId": "asset-ref"}}
    )
    assert request.reference_asset_id == "asset-ref"

def test_sdxl_use_ip_adapter_only_for_text_with_reference():
    use = SdxlDiffusersAdapter._use_ip_adapter
    # IP-Adapter runs on the T2I pipeline with a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id="a")) is True
    # Not for edit jobs (reference + img2img together is a future enhancement)...
    assert use(SimpleNamespace(mode="edit_image", reference_asset_id="a")) is False
    # ...and not without a reference image.
    assert use(SimpleNamespace(mode="text_to_image", reference_asset_id=None)) is False

def test_sdxl_ip_adapter_scale_default_and_clamp():
    adapter = SdxlDiffusersAdapter()
    # Plus-face default is 0.7 — identity-leaning while letting the prompt steer.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={})) == 0.7
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 0.4})) == 0.4
    # Clamped to [0, 1]; unparseable falls back to the default.
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": 5})) == 1.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": -2})) == 0.0
    assert adapter._ip_adapter_scale(SimpleNamespace(advanced={"ipAdapterScale": "x"})) == 0.7

def test_sdxl_reference_run_pipeline_passes_ip_adapter_image(tmp_path, monkeypatch):
    """A T2I job with referenceAssetId drives the IP-Adapter branch of _run_pipeline:
    load_reference_image(project_path, reference_asset_id) → ip_adapter_image kwarg, plus
    a per-request set_ip_adapter_scale. Mirrors test_kolors_reference_run_pipeline_passes_ip_adapter_image
    so it runs torch-free in CI."""

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __init__(self):
            self.scales: list[float] = []

        def set_ip_adapter_scale(self, scale):
            self.scales.append(scale)

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen: list[tuple] = []

    def fake_load_reference_image(project_path, reference_asset_id):
        seen.append((project_path, reference_asset_id))
        return FakeImage()

    monkeypatch.setattr(
        "scene_worker.image_adapters.load_reference_image", fake_load_reference_image
    )

    project_path = tmp_path / "project"
    project_path.mkdir()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_image",
                "model": "sdxl",
                "prompt": "a portrait of the character",
                "referenceAssetId": "asset-ref",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {"ipAdapterScale": 0.5},
            }
        }
    )
    pipe = FakePipe()
    result = SdxlDiffusersAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"), pipe, request, 7, project_path
    )
    # The IP-Adapter branch ran: it loaded the reference image (→ ip_adapter_image
    # kwarg) and applied the per-request scale. (filter_call_kwargs only keeps a
    # pipe's *named* params, and FakePipe takes **kwargs, so we assert via the
    # observable side effects rather than last_kwargs — same as the edit test.)
    assert seen == [(project_path, "asset-ref")]
    assert pipe.scales == [0.5]
    assert result is FakeOutput.images[0]

def test_create_image_adapter_routes_realvisxl():
    # RealVisXL is a photoreal SDXL finetune that rides the same adapter; pickers
    # filter by capability flag, not adapter id (sc-2008).
    adapter = create_image_adapter({"payload": {"model": "realvisxl"}})
    assert adapter.__class__.__name__ == "SdxlDiffusersAdapter"
    assert adapter.id == "sdxl_diffusers"

def test_realvisxl_model_target_defaults():
    target = MODEL_TARGETS["realvisxl"]
    # Same adapter + family as sdxl (sdxl-family LoRAs apply); plain photoreal
    # selectable that complements instantid_realvisxl on the same checkpoint.
    assert target["adapter"] == "sdxl_diffusers"
    assert target["family"] == "sdxl"
    assert target["supportsEdit"] is True
    # SDXL defaults: ~30 steps at guidance 7.0, fp16 variant.
    assert target["steps"] == 30
    assert target["guidanceScale"] == 7.0
    assert target["variant"] == "fp16"
    # RealVisXL_V5.0: photoreal SDXL finetune (openrail++, ungated). Shares the
    # HF cache with the InstantID built-in (no duplicate download).
    assert target["repo"] == "SG161222/RealVisXL_V5.0"
    # IP-Adapter block matches sdxl: any SDXL-UNet checkpoint can reuse it.
    assert target["ipAdapter"] == MODEL_TARGETS["sdxl"]["ipAdapter"]

def test_realvisxl_supports_edit():
    assert model_supports_edit("realvisxl") is True

def test_create_image_adapter_routes_sensenova_u1():
    adapter = create_image_adapter({"payload": {"model": "sensenova_u1_8b"}})
    assert adapter.__class__.__name__ == "SenseNovaU1Adapter"
    assert adapter.id == "sensenova_u1"

def test_image_adapter_env_override_selects_sensenova(monkeypatch):
    monkeypatch.setenv("SCENEWORKS_IMAGE_ADAPTER", "sensenova_u1")
    # Env override wins even when the payload names a different family's model.
    adapter = create_image_adapter({"payload": {"model": "z_image_turbo"}})
    assert adapter.__class__.__name__ == "SenseNovaU1Adapter"

def test_sensenova_u1_model_target_defaults():
    target = MODEL_TARGETS["sensenova_u1_8b"]
    assert target["adapter"] == "sensenova_u1"
    assert target["family"] == "sensenova-u1"
    assert target["steps"] == 50
    # Unified model: the base entry supports instruction editing (it2i).
    assert target["supportsEdit"] is True
    assert target["repo"] == "sensenova/SenseNova-U1-8B-MoT"

def test_sensenova_u1_edit_support():
    # Both the base unified model and the distilled fast variant support editing.
    assert model_supports_edit("sensenova_u1_8b") is True
    assert model_supports_edit("sensenova_u1_8b_fast") is True

def test_sensenova_u1_use_reference_gates_on_mode_and_asset():
    """The Character Studio reference path activates only when mode is
    `character_image` AND a referenceAssetId is present. Mutually exclusive with
    `edit_image` (mirrors QwenImageAdapter._use_reference)."""
    from types import SimpleNamespace
    use_ref = SenseNovaU1Adapter._use_reference
    assert use_ref(SimpleNamespace(mode="character_image", reference_asset_id="asset_x")) is True
    # No reference asset → falls back to plain t2i, never to the edit path with a None source.
    assert use_ref(SimpleNamespace(mode="character_image", reference_asset_id=None)) is False
    # Edit mode is its own branch; character_image gating must not catch it.
    assert use_ref(SimpleNamespace(mode="edit_image", reference_asset_id="asset_x")) is False
    # Plain text-to-image with a stray referenceAssetId stays t2i.
    assert use_ref(SimpleNamespace(mode="text_to_image", reference_asset_id="asset_x")) is False

def test_sensenova_u1_image_guidance_scale_default_pulls_for_character_image():
    """The image-conditioning guidance defaults to 1.0 for edit_image (upstream
    default) and 1.5 for character_image (pull harder toward the reference).
    Per-request overrides via advanced.imageGuidanceScale win in both modes."""
    from types import SimpleNamespace
    img_cfg = SenseNovaU1Adapter._image_guidance_scale
    # Default per-mode:
    assert img_cfg(SimpleNamespace(advanced={})) == 1.0  # edit baseline
    assert img_cfg(SimpleNamespace(advanced={}), default=1.5) == 1.5  # character_image default
    # Override wins regardless of the default:
    assert img_cfg(SimpleNamespace(advanced={"imageGuidanceScale": 2.5}), default=1.5) == 2.5
    # Unparseable values fall back to the supplied default.
    assert img_cfg(SimpleNamespace(advanced={"imageGuidanceScale": "x"}), default=1.5) == 1.5

def test_sensenova_u1_angle_set_loops_augmented_prompts(tmp_path, monkeypatch):
    """SenseNova angle sets must generate one image per canonical angle using
    the per-angle prompt augment. The practical Character Studio picker exposes
    the fast target, but both targets share this adapter path."""
    from scene_worker import image_adapters as ia
    from scene_worker.character_studio_angles import ANGLE_PROMPT_AUGMENTS

    class FakeTorch:
        pass

    adapter = SenseNovaU1Adapter()
    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )
    monkeypatch.setattr(ia, "require_inference_backend_for_gpu_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "select_torch_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ia, "activate_torch_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "select_torch_dtype", lambda *args, **kwargs: "float32")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "load_reference_image", lambda *args, **kwargs: Image.new("RGB", (8, 8)))
    monkeypatch.setattr(adapter, "_load_model", lambda *args, **kwargs: (object(), object()))

    captured: list[dict[str, Any]] = []

    def fake_run_edit(
        torch,
        model,
        tokenizer,
        prompt,
        source_image,
        width,
        height,
        steps,
        guidance_scale,
        img_guidance_scale,
        timestep_shift,
        seed,
    ):
        captured.append(
            {
                "prompt": prompt,
                "sourceSize": source_image.size,
                "steps": steps,
                "guidanceScale": guidance_scale,
                "seed": seed,
            }
        )
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(adapter, "_run_edit_inference", fake_run_edit)

    writer_capture: dict[str, Any] = {}

    def fake_writer(self, *, image_count, image_at_index, raw_settings, **kwargs):
        writer_capture["image_count"] = image_count
        writer_capture["raw_settings"] = raw_settings
        for index in range(image_count):
            image_at_index(index)
        return {"images": [], "count": image_count}

    monkeypatch.setattr(ImageAssetWriter, "write_incremental_outputs", fake_writer)

    job = {
        "id": "job-sensenova-angle",
        "payload": {
            "projectId": "p",
            "mode": "character_image",
            "model": "sensenova_u1_8b_fast",
            "prompt": "the character",
            "referenceAssetId": "ref-1",
            "count": 1,
            "seed": 42,
            "width": 1024,
            "height": 1024,
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

    assert writer_capture["image_count"] == len(CHARACTER_ANGLE_SET_ORDER)
    assert writer_capture["raw_settings"]["angleSet"] is True
    assert writer_capture["raw_settings"]["numInferenceSteps"] == 8
    assert len(captured) == len(CHARACTER_ANGLE_SET_ORDER)
    assert {entry["seed"] for entry in captured} == {42}
    for entry, angle in zip(captured, CHARACTER_ANGLE_SET_ORDER):
        assert ANGLE_PROMPT_AUGMENTS[angle] in entry["prompt"]
        assert entry["sourceSize"] == (2048, 2048)
        assert entry["steps"] == 8
        assert entry["guidanceScale"] == 1.0

def test_sensenova_u1_advertises_character_image_capability():
    """Both sensenova_u1 targets must advertise `character_image` in the builtin
    manifest now that the worker dispatches it through the it2i_generate path
    (sc-2016). Without this the sc-2018 reverse-drift guard would mark the
    engine as 'wired but unreachable from the picker'."""
    manifest = _load_builtin_models_manifest()
    by_id = {model["id"]: model for model in manifest.get("models", [])}
    for model_id in ("sensenova_u1_8b", "sensenova_u1_8b_fast"):
        capabilities = by_id[model_id].get("capabilities") or []
        assert "character_image" in capabilities, (
            f"{model_id} must advertise character_image in the builtin manifest "
            f"so the Image Studio picker surfaces the SenseNova reference flow."
        )
        ui = by_id[model_id].get("ui") or {}
        # sc-2018 audit accepts ui.variationStrength as the edit-backbone engine
        # declaration. SenseNova has no IP-Adapter / face-ID engine block, so
        # this is the required gate for the character_image capability to be honest.
        assert ui.get("variationStrength"), (
            f"{model_id} declares character_image but no ui.variationStrength; "
            f"the sc-2018 audit will fail (engine declaration missing)."
        )
    base_angles = by_id["sensenova_u1_8b"].get("ui", {}).get("viewAngles") or []
    assert base_angles == [], "The 50-step base model must not be advertised for Character Studio angle sets."
    fast_angle_ids = {
        angle["id"] for angle in by_id["sensenova_u1_8b_fast"].get("ui", {}).get("viewAngles") or []
    }
    assert fast_angle_ids == set(CHARACTER_ANGLE_SET_ORDER), (
        "sensenova_u1_8b_fast must expose the canonical angle set in Character Studio; "
        f"missing={set(CHARACTER_ANGLE_SET_ORDER) - fast_angle_ids}, "
        f"extra={fast_angle_ids - set(CHARACTER_ANGLE_SET_ORDER)}"
    )

def test_sensenova_u1_vqa_strips_reasoning():
    strip = SenseNovaU1Adapter._strip_reasoning
    # A complete think block is removed, leaving only the answer.
    assert strip("<think>\nweigh the options\n</think>\n\nIt's nighttime in Paris.") == "It's nighttime in Paris."
    # A dangling/unclosed think block (reasoning truncated by the token cap) yields no leak.
    assert strip("<think>\nreasoning that got cut off mid-thought") == ""
    # A plain answer (no reasoning) is returned unchanged.
    assert strip("A woman holds a yellow umbrella.") == "A woman holds a yellow umbrella."

def test_sensenova_u1_vqa_requires_question():
    # The VQA entry point validates the question before any model load (no torch).
    job = {
        "id": "job_vqa",
        "payload": {"projectId": "p", "sourceAssetId": "asset_1", "question": "   ", "model": "sensenova_u1_8b"},
    }
    noop = lambda *args, **kwargs: None  # noqa: E731
    try:
        SenseNovaU1Adapter().answer_question(settings=None, job=job, progress=noop, cancel_requested=lambda: False)
    except RuntimeError as exc:
        assert "requires a question" in str(exc)
    else:
        raise AssertionError("VQA must reject an empty question.")

def test_interleave_resolution_for_snaps_to_interleave_buckets():
    # Interleave uses its own (smaller) trained buckets, distinct from t2i.
    assert interleave_resolution_for(1000, 1000) == (1536, 1536)
    assert interleave_resolution_for(1920, 1080) == (2048, 1152)
    assert interleave_resolution_for(1080, 1920) == (1152, 2048)
    # Same 16:9 request snaps differently for interleave vs t2i.
    assert interleave_resolution_for(1920, 1080) != sensenova_resolution_for(1920, 1080)

def test_generate_interleaved_requires_prompt():
    # The prompt is validated before any model load (no torch needed).
    adapter = SenseNovaU1Adapter()
    job = {"id": "job_il", "payload": {"projectId": "p", "prompt": "   "}}
    noop = lambda *args, **kwargs: None  # noqa: E731
    with pytest.raises(RuntimeError, match="requires a prompt"):
        adapter.generate_interleaved(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )

def test_generate_interleaved_rejects_non_sensenova_model():
    adapter = SenseNovaU1Adapter()
    job = {"id": "job_il", "payload": {"projectId": "p", "prompt": "guide", "model": "z_image_turbo"}}
    noop = lambda *args, **kwargs: None  # noqa: E731
    with pytest.raises(RuntimeError, match="not a SenseNova-U1 target"):
        adapter.generate_interleaved(
            settings=None, job=job, request=image_request_from_job(job), project_path=None,
            progress=noop, cancel_requested=lambda: False,
        )

def test_interleave_segments_interleave_text_and_images():
    assets = [
        {"assetId": "asset_a", "mediaPath": "assets/images/a.png"},
        {"assetId": "asset_b", "mediaPath": "assets/images/b.png"},
    ]
    text = "Boil the water.<image>Steep the leaves.<image>Pour and enjoy."
    segments = SenseNovaU1Adapter._build_interleaved_segments(text, assets)
    assert segments == [
        {"type": "text", "text": "Boil the water."},
        {"type": "image", "assetId": "asset_a", "path": "assets/images/a.png"},
        {"type": "text", "text": "Steep the leaves."},
        {"type": "image", "assetId": "asset_b", "path": "assets/images/b.png"},
        {"type": "text", "text": "Pour and enjoy."},
    ]

def test_interleave_segments_handles_leading_image_and_blank_text():
    assets = [{"assetId": "asset_a", "mediaPath": "assets/images/a.png"}]
    segments = SenseNovaU1Adapter._build_interleaved_segments("<image>Only a caption.", assets)
    assert segments == [
        {"type": "image", "assetId": "asset_a", "path": "assets/images/a.png"},
        {"type": "text", "text": "Only a caption."},
    ]

def test_interleave_segments_caps_images_to_available_assets():
    # More <image> markers than generated images: extra markers emit no image segment.
    assets = [{"assetId": "asset_a", "mediaPath": "assets/images/a.png"}]
    segments = SenseNovaU1Adapter._build_interleaved_segments("A<image>B<image>C", assets)
    assert segments == [
        {"type": "text", "text": "A"},
        {"type": "image", "assetId": "asset_a", "path": "assets/images/a.png"},
        {"type": "text", "text": "B"},
        {"type": "text", "text": "C"},
    ]

def test_write_interleaved_document_persists_document_and_image_assets(tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {"id": "job-1", "payload": {"projectId": "project-1", "model": "sensenova_u1_8b", "prompt": "guide"}}
    progress_results = []

    def progress(status, stage, value, message, result=None):
        if result is not None:
            progress_results.append(result)

    result = SenseNovaU1Adapter()._write_interleaved_document(
        project_path=project_path,
        request=image_request_from_job(job),
        job=job,
        project_id="project-1",
        model_id="sensenova_u1_8b",
        prompt="An illustrated guide to brewing tea",
        seed=7,
        generated_text="Boil water.<image>Steep.<image>Done.",
        images=[Image.new("RGB", (16, 16), (255, 0, 0)), Image.new("RGB", (16, 16), (0, 255, 0))],
        cancel_requested=lambda: False,
        progress=progress,
        raw_settings={"maxImages": 6, "resolution": "2048x1152"},
    )

    document_dir = project_path / "assets" / "documents"
    doc_jsons = [p for p in document_dir.glob("*.json") if not p.name.endswith(".sceneworks.json")]
    # The worker writes the document body; the Rust API builds the sidecar from
    # the document fact (story 1656 slice 4), so no sidecar lands here.
    assert len(doc_jsons) == 1
    assert not list(document_dir.glob("*.sceneworks.json"))

    document = json.loads(doc_jsons[0].read_text(encoding="utf-8"))
    assert [segment["type"] for segment in document["segments"]] == ["text", "image", "text", "image", "text"]
    assert document["model"] == "sensenova_u1_8b"
    assert document["jobId"] == "job-1"

    # The two generated images persist as ordinary image assets (Rust writes their
    # sidecars from facts); the worker still saves their PNG bytes, nested in the
    # generation-set subfolder.
    assert len(list((project_path / "assets" / "images").rglob("*.png"))) == 2
    assert len(result["imageAssetIds"]) == 2
    assert result["documentId"] == document["id"]
    assert result["segments"] == document["segments"]
    assert progress_results and progress_results[-1]["documentId"] == document["id"]

    # The worker emits flat facts: two image facts + one document fact; the Rust
    # API builds + indexes all three sidecars on completion.
    asset_writes = result["assetWrites"]
    assert len(asset_writes) == 3
    assert all(write["mimeType"] == "image/png" for write in asset_writes[:2])
    document_write = asset_writes[-1]
    assert document_write["type"] == "document"
    assert document_write["assetId"] == result["documentAssetId"]
    assert document_write["mediaPath"] == f"assets/documents/{document['id']}.json"
    assert document_write["mimeType"] == "application/json"
    assert document_write["mode"] == "interleave"
    assert document_write["imageCount"] == 2
    assert document_write["parents"] == result["imageAssetIds"]

def test_sensenova_resolution_for_snaps_to_buckets():
    # Snaps any request to the nearest SenseNova-U1 trained bucket by aspect ratio.
    assert sensenova_resolution_for(1024, 1024) == (2048, 2048)
    assert sensenova_resolution_for(1920, 1080) == (2720, 1536)
    assert sensenova_resolution_for(1080, 1920) == (1536, 2720)
    assert sensenova_resolution_for(1024, 768) == (2368, 1760)
    assert sensenova_resolution_for(768, 1024) == (1760, 2368)
    assert sensenova_resolution_for(1500, 1000) == (2496, 1664)

def test_image_request_allows_sensenova_wide_buckets():
    # The W/H clamp was raised (2048 -> 4096) so SenseNova's true trained buckets
    # (up to 3456) survive into the adapter, which snaps by the requested aspect
    # ratio. Truncating 2720x1536 to 2048x1536 would mis-snap 16:9 to a ~4:3 bucket.
    request = image_request_from_job({"payload": {"projectId": "p", "width": 2720, "height": 1536}})
    assert (request.width, request.height) == (2720, 1536)
    assert sensenova_resolution_for(request.width, request.height) == (2720, 1536)
    assert sensenova_resolution_for(2048, 1536) != (2720, 1536)

def test_image_request_clamps_above_new_ceiling():
    request = image_request_from_job({"payload": {"projectId": "p", "width": 9000, "height": 9000}})
    assert request.width == 4096
    assert request.height == 4096

def test_image_request_parses_optional_upscale_contract():
    default_request = image_request_from_job({"payload": {"projectId": "p"}})
    assert default_request.upscale.enabled is False
    assert default_request.upscale.factor == 2
    assert default_request.upscale.engine == "real-esrgan"

    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 4, "engine": "real-esrgan"},
            }
        }
    )
    assert request.upscale.enabled is True
    assert request.upscale.factor == 4
    assert request.upscale.engine == "real-esrgan"

def test_create_image_upscaler_rejects_unknown_engine():
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 2, "engine": "mystery-upscale"},
            }
        }
    )

    with pytest.raises(RuntimeError, match="Unsupported image upscale engine"):
        create_image_upscaler(request)

def test_create_image_upscaler_accepts_aurasr_engine():
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 4, "engine": "aura-sr"},
            }
        }
    )

    upscaler = create_image_upscaler(request, settings=SimpleNamespace(gpu_id="cpu"))

    assert isinstance(upscaler, AuraSrUpscaler)
    assert upscaler.id == "aura-sr"

def test_real_esrgan_resolves_manifest_weight_through_hf_cache(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    resolved_path = tmp_path / "RealESRGAN_x2plus.pth"
    resolved_path.write_bytes(b"weights")

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise RuntimeError("cache miss")
        return str(resolved_path)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    settings = SimpleNamespace(data_dir=tmp_path / "data", gpu_id="cpu")
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 2, "engine": "real-esrgan"},
                "modelManifestEntry": {
                    "resources": {
                        "imageUpscalers": {
                            "real-esrgan": {
                                "x2": {"repo": "example/upscalers", "file": "RealESRGAN_x2plus.pth"}
                            }
                        }
                    }
                },
            }
        }
    )

    path = RealEsrganUpscaler(settings=settings)._resolve_model_path(
        request,
        REAL_ESRGAN_MODEL_SPECS[2],
    )

    assert path == resolved_path
    assert calls[0]["repo_id"] == "example/upscalers"
    assert calls[0]["filename"] == "RealESRGAN_x2plus.pth"
    assert calls[0]["cache_dir"] == str(settings.data_dir / "cache" / "huggingface" / "hub")
    assert calls[0]["local_files_only"] is True
    assert calls[-1].get("local_files_only") is not True

def test_aurasr_resolves_manifest_snapshot_through_hf_cache(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "model.safetensors").write_bytes(b"weights")
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise RuntimeError("cache miss")
        return str(snapshot_dir)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    settings = SimpleNamespace(data_dir=tmp_path / "data", gpu_id="cpu")
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 4, "engine": "aura-sr"},
                "modelManifestEntry": {
                    "resources": {
                        "imageUpscalers": {
                            "aura-sr": {
                                "x4": {"repo": "example/aura-sr", "file": "model.safetensors"}
                            }
                        }
                    }
                },
            }
        }
    )

    path = AuraSrUpscaler(settings=settings)._resolve_model_path(request)

    assert path == snapshot_dir / "model.safetensors"
    assert calls[0]["repo_id"] == "example/aura-sr"
    assert calls[0]["allow_patterns"] == ["model.safetensors", "config.json", "LICENSE.md", "README.md"]
    assert calls[0]["cache_dir"] == str(settings.data_dir / "cache" / "huggingface" / "hub")
    assert calls[0]["local_files_only"] is True
    assert calls[-1].get("local_files_only") is not True

def test_aurasr_upscaler_rejects_non_4x_request(monkeypatch):
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 2, "engine": "aura-sr"},
            }
        }
    )

    with pytest.raises(RuntimeError, match="supports only 4x"):
        AuraSrUpscaler(settings=SimpleNamespace(gpu_id="cpu")).upscale(
            Image.new("RGB", (2, 2), "white"),
            request=request,
            cancel_requested=lambda: False,
        )

def test_aurasr_upscaler_caches_loaded_model_and_threads_batch_size(tmp_path, monkeypatch):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"stub")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            mps = None

    seen: dict[str, Any] = {"loads": 0, "calls": []}

    class FakeUpsampler:
        def __init__(self):
            self.device = None
            self.evaluated = False

        def to(self, device):
            self.device = device

        def eval(self):
            self.evaluated = True

    class FakeAuraModel:
        def __init__(self):
            self.upsampler = FakeUpsampler()

        def upscale_4x_overlapped(self, image, max_batch_size=16):
            seen["calls"].append(max_batch_size)
            return image.resize((image.width * 4, image.height * 4))

    class FakeAuraSR:
        @staticmethod
        def from_pretrained(model_id, use_safetensors=True):
            seen["loads"] += 1
            seen["model_id"] = model_id
            seen["use_safetensors"] = use_safetensors
            model = FakeAuraModel()
            seen["upsampler"] = model.upsampler
            return model

    fake_aura_module = SimpleNamespace(AuraSR=FakeAuraSR)
    imports: list[str] = []

    def fake_import_module(name):
        imports.append(name)
        if name == "torch":
            return FakeTorch
        if name == "aura_sr":
            return fake_aura_module
        return importlib.import_module(name)

    monkeypatch.setattr("scene_worker.image_adapters.importlib.import_module", fake_import_module)
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "p",
                "upscale": {"enabled": True, "factor": 4, "engine": "aura-sr"},
                "advanced": {"auraSrModelPath": str(weights), "auraSrMaxBatchSize": 3},
            }
        }
    )
    upscaler = AuraSrUpscaler(settings=SimpleNamespace(gpu_id="cpu"))

    first = upscaler.upscale(Image.new("RGB", (3, 4), "white"), request=request, cancel_requested=lambda: False)
    second = upscaler.upscale(Image.new("RGB", (2, 5), "white"), request=request, cancel_requested=lambda: False)

    assert first.size == (12, 16)
    assert second.size == (8, 20)
    assert imports == ["torch", "aura_sr"]
    assert seen["loads"] == 1
    assert seen["model_id"] == str(weights)
    assert seen["use_safetensors"] is True
    assert seen["calls"] == [3, 3]
    assert seen["upsampler"].device == "cpu"
    assert seen["upsampler"].evaluated is True

def test_image_asset_writer_retains_original_and_adds_upscaled_variant(monkeypatch, tmp_path):
    class FakeUpscaler:
        id = "real-esrgan"

        def upscale(self, image, *, request, cancel_requested):
            assert request.upscale.factor == 2
            assert cancel_requested() is False
            return image.resize((image.width * 2, image.height * 2))

    monkeypatch.setattr("scene_worker.image_adapters.create_image_upscaler", lambda *_args, **_kwargs: FakeUpscaler())
    project_path = tmp_path / "project"
    project_path.mkdir()
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "z_image_turbo",
            "count": 1,
            "width": 16,
            "height": 16,
            "upscale": {"enabled": True, "factor": 2, "engine": "real-esrgan"},
        },
    }

    result = ImageAssetWriter().write_incremental_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        image_count=1,
        image_at_index=lambda _index: Image.new("RGB", (16, 12), "navy"),
        adapter_id="z_image_diffusers",
        progress=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
        job_id=job["id"],
    )

    assert result["expectedCount"] == 2
    assert len(result["assetWrites"]) == 2

    original_write, upscaled_write = result["assetWrites"]
    assert (original_write["width"], original_write["height"]) == (16, 12)
    assert "upscale" not in original_write["rawAdapterSettings"]
    assert upscaled_write["sourceAssetId"] == original_write["assetId"]
    assert upscaled_write["parents"] == [original_write["assetId"]]
    assert upscaled_write["extra"] == {
        "isUpscaled": True,
        "upscaledFromAssetId": original_write["assetId"],
        "factor": 2,
        "engine": "real-esrgan",
    }
    assert (upscaled_write["width"], upscaled_write["height"]) == (32, 24)
    assert upscaled_write["rawAdapterSettings"]["upscale"] == {
        "enabled": True,
        "engine": "real-esrgan",
        "factor": 2,
        "sourceWidth": 16,
        "sourceHeight": 12,
        "width": 32,
        "height": 24,
    }
    with Image.open(project_path / original_write["mediaPath"]) as saved:
        assert saved.size == (16, 12)
    with Image.open(project_path / upscaled_write["mediaPath"]) as saved:
        assert saved.size == (32, 24)

def test_create_image_adapter_routes_sensenova_u1_fast():
    adapter = create_image_adapter({"payload": {"model": "sensenova_u1_8b_fast"}})
    assert adapter.__class__.__name__ == "SenseNovaU1Adapter"
    assert adapter.id == "sensenova_u1"

def test_sensenova_u1_fast_model_target_defaults():
    target = MODEL_TARGETS["sensenova_u1_8b_fast"]
    assert target["adapter"] == "sensenova_u1"
    assert target["family"] == "sensenova-u1"
    # 8-step distill LoRA: shares the base weights, cfg 1.0 / 8 steps.
    assert target["steps"] == 8
    assert target["guidanceScale"] == 1.0
    # Distilled editing (it2i) reuses the same variant + distill LoRA.
    assert target["supportsEdit"] is True
    assert target["repo"] == "sensenova/SenseNova-U1-8B-MoT"
    assert target["distillLora"] == {
        "repo": "sensenova/SenseNova-U1-8B-MoT-LoRAs",
        "file": "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors",
    }

def test_sensenova_u1_guidance_scale_defaults_from_model_target():
    request = image_request_from_job({"payload": {"projectId": "project_x", "prompt": "a cat"}})
    # The fast variant defaults to cfg 1.0; the base variant to cfg 4.0.
    assert SenseNovaU1Adapter._guidance_scale(request, MODEL_TARGETS["sensenova_u1_8b_fast"]) == 1.0
    assert SenseNovaU1Adapter._guidance_scale(request, MODEL_TARGETS["sensenova_u1_8b"]) == 4.0
    # An explicit request value overrides the per-model default.
    override = image_request_from_job(
        {"payload": {"projectId": "project_x", "prompt": "a cat", "advanced": {"guidanceScale": 2.5}}}
    )
    assert SenseNovaU1Adapter._guidance_scale(override, MODEL_TARGETS["sensenova_u1_8b_fast"]) == 2.5

def test_huggingface_repo_cache_path_stays_under_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    path = huggingface_repo_cache_path(r"..\outside/../../model")

    assert path is not None
    path.relative_to((tmp_path / "hub").resolve())
    assert path.name.startswith("models--")

def test_repo_cache_path_is_a_single_guarded_helper(monkeypatch, tmp_path):
    # The lora adapter previously had its own unguarded copy with a different
    # cache-root order. Every adapter must now resolve through the one guarded
    # helper, so a traversal repo id can't escape the cache root from any entry point.
    from scene_worker.hf_cache import huggingface_repo_cache_path as shared
    from scene_worker.lora_adapters import huggingface_repo_cache_path as lora_entry

    assert huggingface_repo_cache_path is shared
    assert lora_entry is shared

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    path = lora_entry(r"..\..\outside/model")
    assert path is not None
    path.relative_to((tmp_path / "hub").resolve())
    assert path.name.startswith("models--")

def test_image_asset_writer_reports_partial_result_assets(tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "z_image_turbo",
            "count": 2,
            "width": 16,
            "height": 16,
        },
    }
    progress_calls = []

    def progress(status, stage, value, message, result=None):
        progress_calls.append(
            {
                "status": status,
                "stage": stage,
                "value": value,
                "message": message,
                "result": result,
            }
        )

    result = ImageAssetWriter().write_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        images=[
            Image.new("RGB", (16, 16), (255, 0, 0)),
            Image.new("RGB", (16, 16), (0, 255, 0)),
        ],
        adapter_id="procedural_preview",
        progress=progress,
        cancel_requested=lambda: False,
        raw_settings={"preview": True},
    )

    result_progress = [call["result"] for call in progress_calls if call["result"]]
    # The worker reports flat facts now (story 1656); the Rust API builds the
    # sidecars and injects assets/assetIds. So the worker-side result streams the
    # growing assetWrites list, one more entry per image.
    assert [len(item["assetWrites"]) for item in result_progress] == [1, 2]
    assert result_progress[0]["expectedCount"] == 2
    assert result_progress[0]["generationSetId"] == result["generationSetId"]
    assert result_progress[0]["assetWrites"][0]["mediaPath"].startswith("assets/images/")
    assert [write["assetId"] for write in result_progress[1]["assetWrites"]] == [
        write["assetId"] for write in result["assetWrites"]
    ]
    assert result["expectedCount"] == 2

def test_image_asset_writer_persists_each_image_before_requesting_next(tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "z_image_turbo",
            "count": 2,
            "width": 16,
            "height": 16,
        },
    }

    def image_at_index(index):
        if index == 1:
            # The worker saves each PNG before requesting the next image (so a
            # multi-image batch streams). Sidecars are written by Rust now (story
            # 1656), so only the PNG lands on disk here — rglob into the
            # per-generation-set subfolder.
            assert len(list((project_path / "assets" / "images").rglob("*.png"))) == 1
        return Image.new("RGB", (16, 16), (255, 0, 0) if index == 0 else (0, 255, 0))

    result = ImageAssetWriter().write_incremental_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        image_count=2,
        image_at_index=image_at_index,
        adapter_id="z_image_diffusers",
        progress=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
    )

    assert len(result["assetWrites"]) == 2
    assert len(list((project_path / "assets" / "images").rglob("*.png"))) == 2

def test_image_asset_writer_does_not_clobber_identical_jobs(tmp_path):
    # Two jobs sharing the same date + model + prompt + image index must not
    # collide on disk: the per-generation-set subfolder keeps each job's PNG
    # distinct so the first asset's pixels are never overwritten by the second.
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "sensenova_u1_8b",
            "count": 1,
            "width": 16,
            "height": 16,
        },
    }

    def run(color):
        return ImageAssetWriter().write_incremental_outputs(
            request=image_request_from_job(job),
            project_path=project_path,
            image_count=1,
            image_at_index=lambda _index: Image.new("RGB", (16, 16), color),
            adapter_id="sensenova_u1",
            progress=lambda *_args, **_kwargs: None,
            cancel_requested=lambda: False,
            raw_settings={"realModelInference": True},
        )

    first = run((255, 0, 0))
    second = run((0, 255, 0))

    # Distinct generation sets, distinct asset ids, distinct files on disk.
    assert first["generationSetId"] != second["generationSetId"]
    assert [write["assetId"] for write in first["assetWrites"]] != [
        write["assetId"] for write in second["assetWrites"]
    ]
    pngs = list((project_path / "assets" / "images").rglob("*.png"))
    assert len(pngs) == 2

    # The first asset's recorded path still points at the first job's pixels;
    # the second job did not overwrite it.
    first_path = project_path / first["assetWrites"][0]["mediaPath"]
    second_path = project_path / second["assetWrites"][0]["mediaPath"]
    assert first_path.exists() and second_path.exists()
    assert first_path != second_path
    with Image.open(first_path) as handle:
        assert handle.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    with Image.open(second_path) as handle:
        assert handle.convert("RGB").getpixel((0, 0)) == (0, 255, 0)

def test_image_asset_writer_records_actual_output_dimensions(tmp_path):
    # The reported width/height (which Rust writes into the sidecar's file block)
    # must reflect the PNG actually saved (e.g. a model that snaps to a trained
    # bucket), not the requested size or the old min(request, 1280) cap.
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "sensenova_u1_8b",
            "count": 1,
            "width": 2720,
            "height": 1536,
        },
    }

    result = ImageAssetWriter().write_incremental_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        image_count=1,
        image_at_index=lambda _index: Image.new("RGB", (2720, 1536), "navy"),
        adapter_id="sensenova_u1",
        progress=lambda *_args, **_kwargs: None,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
    )

    write = result["assetWrites"][0]
    assert (write["width"], write["height"]) == (2720, 1536)

def test_image_asset_writer_batch_progress_is_monotonic(tmp_path):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )
    job = {
        "id": "job-1",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Neon alley",
            "model": "z_image_turbo",
            "count": 4,
            "width": 16,
            "height": 16,
        },
    }
    progress_values = []

    def progress(_status, _stage, value, _message, result=None):
        progress_values.append(value)

    def image_at_index(index):
        progress("running", "generating", image_batch_progress(index, 4), f"Running image {index + 1} of 4.")
        return Image.new("RGB", (16, 16), (255, 0, 0))

    ImageAssetWriter().write_incremental_outputs(
        request=image_request_from_job(job),
        project_path=project_path,
        image_count=4,
        image_at_index=image_at_index,
        adapter_id="z_image_diffusers",
        progress=progress,
        cancel_requested=lambda: False,
        raw_settings={"realModelInference": True},
    )

    assert progress_values == sorted(progress_values)

def test_keepalive_heartbeat_reports_empty_models_when_source_is_none(monkeypatch):
    calls = []

    class StopAfterOneHeartbeat:
        def wait(self, _interval):
            return bool(calls)

    class Settings:
        worker_id = "worker-1"
        heartbeat_seconds = 1

    def capture_heartbeat(_api, _settings, _status, _current_job_id=None, loaded_models=None):
        calls.append(loaded_models)

    monkeypatch.setattr("scene_worker.runtime.heartbeat", capture_heartbeat)

    keep_job_alive(
        api=object(),
        settings=Settings(),
        job_id="job-1",
        status="busy",
        stop_event=StopAfterOneHeartbeat(),
        loaded_models=None,
    )

    assert calls == [[]]

def test_format_batch_running_message_names_completed_count_after_first_image():
    assert format_batch_running_message("Z-Image", 0, 4) == "Running Z-Image 1 of 4."
    assert format_batch_running_message("Z-Image", 2, 4) == "Generated 2 of 4. Running Z-Image 3 of 4."

def test_gpu_memory_snapshot_returns_none_when_cuda_unavailable():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    assert gpu_memory_snapshot(Torch, "cuda:0") is None
    assert gpu_memory_snapshot(Torch, "cpu") is None

def test_gpu_memory_snapshot_reports_allocated_and_reserved_bytes():
    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def memory_allocated(index=None):
                return 50 * 1024 * 1024

            @staticmethod
            def memory_reserved(index=None):
                return 60 * 1024 * 1024

    snapshot = gpu_memory_snapshot(Torch, "cuda:0")

    assert snapshot == {"device": "cuda:0", "allocatedMb": 50.0, "reservedMb": 60.0}

def test_upscaler_tile_slices_cover_edges_without_overlap_gaps():
    assert tile_slices(5, 3, 2) == [
        TileSlice(0, 0, 2, 2),
        TileSlice(2, 0, 4, 2),
        TileSlice(4, 0, 5, 2),
        TileSlice(0, 2, 2, 3),
        TileSlice(2, 2, 4, 3),
        TileSlice(4, 2, 5, 3),
    ]
    assert tile_slices(5, 3, 0) == [TileSlice(0, 0, 5, 3)]

def test_real_esrgan_upscale_lazily_imports_torch_and_reuses_device_helpers(tmp_path, monkeypatch):
    weights = tmp_path / "realesrgan.pth"
    weights.write_bytes(b"stub")

    class FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        float16 = "float16"

        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            mps = None

    imports: list[str] = []

    def fake_import_module(name):
        imports.append(name)
        if name == "torch":
            return FakeTorch
        return importlib.import_module(name)

    seen: dict[str, Any] = {}

    def fake_load_model(self, torch, weights_path, *, factor, device, dtype):
        seen.update({"torch": torch, "weights_path": weights_path, "factor": factor, "device": device, "dtype": dtype})
        return object()

    def fake_upscale_with_model(self, torch, model, image, *, factor, device, dtype, tile_size, tile_pad):
        seen.update({"tile_size": tile_size, "tile_pad": tile_pad})
        return image.resize((image.width * factor, image.height * factor))

    monkeypatch.setattr("scene_worker.upscalers.importlib.import_module", fake_import_module)
    monkeypatch.setattr(RealESRGANUpscaler, "_load_model", fake_load_model)
    monkeypatch.setattr(RealESRGANUpscaler, "_upscale_with_model", fake_upscale_with_model)

    result = RealESRGANUpscaler().upscale(
        Image.new("RGB", (3, 4), "white"),
        job=UpscaleJob(factor=2, weights_path=weights, tile_size=64, tile_pad=4),
        settings=SimpleNamespace(gpu_id="cpu"),
    )

    assert result.size == (6, 8)
    assert imports == ["torch"]
    assert seen == {
        "torch": FakeTorch,
        "weights_path": weights,
        "factor": 2,
        "device": "cpu",
        "dtype": "float32",
        "tile_size": 64,
        "tile_pad": 4,
    }

def test_pipeline_component_devices_inspects_known_submodules():
    class Module:
        def __init__(self, device):
            self.device = device

    class Pipe:
        components = {"unet": Module("cuda:0"), "text_encoder": Module("cuda:0"), "vae": Module("cpu")}

    assert pipeline_component_devices(Pipe()) == ["cpu", "cuda:0"]

def test_pipeline_component_devices_falls_back_to_named_attributes():
    class Module:
        def __init__(self, device):
            self.device = device

    class Pipe:
        transformer = Module("cuda:1")
        vae = Module("cuda:1")

    assert pipeline_component_devices(Pipe()) == ["cuda:1"]

def test_verify_pipeline_on_device_raises_when_components_stayed_on_cpu():
    class Module:
        device = "cpu"

    class Pipe:
        components = {"unet": Module(), "vae": Module()}

    with pytest.raises(RuntimeError, match="did not move onto cuda:0"):
        verify_pipeline_on_device(
            Pipe(),
            requested_device="cuda:0",
            model_label="Z-Image-Turbo",
            allow_offload=False,
        )

def test_verify_pipeline_on_device_raises_when_any_component_stayed_on_cpu():
    class Module:
        def __init__(self, device):
            self.device = device

    class Pipe:
        components = {"transformer": Module("cuda:0"), "vae": Module("cpu")}

    with pytest.raises(RuntimeError, match="pipeline components are on cpu, cuda:0"):
        verify_pipeline_on_device(
            Pipe(),
            requested_device="cuda:0",
            model_label="Z-Image-Turbo",
            allow_offload=False,
        )

def test_verify_pipeline_on_device_rejects_wrong_cuda_index():
    class Module:
        device = "cuda:10"

    class Pipe:
        components = {"transformer": Module()}

    with pytest.raises(RuntimeError, match="did not move onto cuda:1"):
        verify_pipeline_on_device(
            Pipe(),
            requested_device="cuda:1",
            model_label="Z-Image-Turbo",
            allow_offload=False,
        )

def test_verify_pipeline_on_device_allows_cpu_offload_layouts():
    class Module:
        device = "cpu"

    class Pipe:
        components = {"unet": Module(), "vae": Module()}

    devices = verify_pipeline_on_device(
        Pipe(),
        requested_device="cuda:0",
        model_label="Z-Image-Turbo",
        allow_offload=True,
    )

    assert devices == ["cpu"]

def test_verify_pipeline_on_device_accepts_matching_cuda_index():
    class Module:
        def __init__(self, device):
            self.device = device

    class Pipe:
        components = {"unet": Module("cuda:0"), "vae": Module("cuda:0")}

    devices = verify_pipeline_on_device(
        Pipe(),
        requested_device="cuda:0",
        model_label="Z-Image-Turbo",
        allow_offload=False,
    )

    assert devices == ["cuda:0"]

def test_emit_worker_event_writes_json_to_stdout(capsys):
    emit_worker_event("image_inference_start", jobId="job-1", imageIndex=2)

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["event"] == "image_inference_start"
    assert payload["jobId"] == "job-1"
    assert payload["imageIndex"] == 2
    assert "reportedAt" in payload

def test_z_image_adapter_emits_phase_diagnostics_and_running_message(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )

    class FakeTransformer:
        device = "cuda:0"

        def parameters(self):
            return iter([])

    class FakePipe:
        components = {"transformer": FakeTransformer(), "vae": FakeTransformer()}

        def __init__(self):
            self.calls = 0

        def to(self, device):
            self._device = device
            return self

        def __call__(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(images=[Image.new("RGB", (8, 8), (10, 20, 30))])

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(repo, **_kwargs):
            return FakePipe()

    class FakeDiffusers:
        ZImagePipeline = FakePipelineClass

        @staticmethod
        def __getattr__(name):
            raise AttributeError(name)

    class FakeTorch:
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def memory_allocated(index=None):
                return 10 * 1024 * 1024

            @staticmethod
            def memory_reserved(index=None):
                return 12 * 1024 * 1024

            @staticmethod
            def set_device(_device):
                return None

            @staticmethod
            def empty_cache():
                return None

        class backends:
            mps = None

        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    real_import = __import__

    def fake_import_module(name):
        if name == "torch":
            return FakeTorch
        if name == "diffusers":
            return FakeDiffusers
        return real_import(name)

    monkeypatch.setattr("scene_worker.image_adapters.importlib.import_module", fake_import_module)

    job = {
        "id": "job-z",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Stormy bridge",
            "model": "z_image_turbo",
            "count": 2,
            "width": 16,
            "height": 16,
        },
    }

    progress_calls: list[dict] = []

    def progress(status, stage, value, message, result=None):
        progress_calls.append(
            {"status": status, "stage": stage, "value": value, "message": message, "result": result}
        )

    adapter = ZImageDiffusersAdapter()
    adapter.generate(
        settings=SimpleNamespace(data_dir=data_dir, gpu_id="0"),
        job=job,
        request=image_request_from_job(job),
        project_path=project_path,
        progress=progress,
        cancel_requested=lambda: False,
    )

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines() if line.strip()]
    event_names = [event["event"] for event in events]
    assert "image_pipeline_load_start" in event_names
    assert "image_pipeline_load_complete" in event_names
    assert "image_pipeline_on_device" in event_names
    assert event_names.count("image_inference_start") == 2
    assert event_names.count("image_inference_complete") == 2
    on_device = next(event for event in events if event["event"] == "image_pipeline_on_device")
    assert on_device["componentDevices"] == ["cuda:0"]
    assert on_device["gpuMemory"]["allocatedMb"] == 10.0

    running_messages = [call["message"] for call in progress_calls if call["status"] == "running"]
    assert running_messages == [
        "Running Z-Image 1 of 2.",
        "Generated 1 of 2. Running Z-Image 2 of 2.",
    ]

def test_z_image_adapter_fails_fast_when_pipeline_stays_on_cpu_with_offload_fallback(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    project_path = tmp_path / "project"
    data_dir.mkdir()
    project_path.mkdir()
    (data_dir / "recent-projects.json").write_text(
        json.dumps([{"id": "project-1", "path": str(project_path)}]),
        encoding="utf-8",
    )

    class StuckOnCpu:
        device = "cpu"

    class FakePipe:
        components = {"transformer": StuckOnCpu(), "vae": StuckOnCpu()}

        def to(self, device):
            return self

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(repo, **_kwargs):
            return FakePipe()

    class FakeDiffusers:
        ZImagePipeline = FakePipelineClass

    class FakeTorch:
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def memory_allocated(index=None):
                return 0

            @staticmethod
            def memory_reserved(index=None):
                return 0

            @staticmethod
            def set_device(_device):
                return None

            @staticmethod
            def empty_cache():
                return None

        class backends:
            mps = None

    def fake_import_module(name):
        if name == "torch":
            return FakeTorch
        if name == "diffusers":
            return FakeDiffusers
        raise ImportError(name)

    monkeypatch.setattr("scene_worker.image_adapters.importlib.import_module", fake_import_module)

    job = {
        "id": "job-cpu-stuck",
        "payload": {
            "projectId": "project-1",
            "mode": "text_to_image",
            "prompt": "Misty harbor",
            "model": "z_image_turbo",
            "count": 1,
            "width": 16,
            "height": 16,
            "advanced": {"cpuOffload": True},
        },
    }

    adapter = ZImageDiffusersAdapter()

    with pytest.raises(RuntimeError, match="did not move onto"):
        adapter.generate(
            settings=SimpleNamespace(data_dir=data_dir, gpu_id="0"),
            job=job,
            request=image_request_from_job(job),
            project_path=project_path,
            progress=lambda *_args, **_kwargs: None,
            cancel_requested=lambda: False,
        )

def test_edit_run_pipeline_threads_project_path(tmp_path, monkeypatch):
    """Regression: ZImage/Qwen ._run_pipeline must accept project_path.

    sc-1678's resolver-threading refactor left project_path referenced inside the
    edit_image branch (load_source_image(project_path, request)) without making it
    a parameter — a latent NameError that only fires at real img2img inference,
    which the mocked suite never exercises. KolorsDiffusersAdapter is the reference
    pattern. This guards both the signature and the live edit-branch call site.
    """
    import inspect

    # Signature guard: project_path must precede the optional cancel_requested,
    # because every call site passes it positionally.
    for adapter_cls in (ZImageDiffusersAdapter, QwenImageAdapter, SdxlDiffusersAdapter):
        params = list(inspect.signature(adapter_cls._run_pipeline).parameters)
        assert "project_path" in params, f"{adapter_cls.__name__}._run_pipeline must accept project_path"
        assert params.index("project_path") < params.index("cancel_requested")

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    seen_paths: list[Path] = []

    def fake_load_source_image(project_path, _request):
        seen_paths.append(project_path)
        return FakeImage()

    monkeypatch.setattr("scene_worker.image_adapters.load_source_image", fake_load_source_image)

    project_path = tmp_path / "project"
    project_path.mkdir()

    # Runtime guard: invoke the edit_image branch directly (gpu_id="cpu" keeps
    # device selection off real torch) so project_path resolves at the exact line
    # that raised NameError instead of crashing.
    for adapter, model in (
        (ZImageDiffusersAdapter(), "z_image_edit"),
        (QwenImageAdapter(), "qwen_image_edit"),
        (SdxlDiffusersAdapter(), "sdxl"),
    ):
        seen_paths.clear()
        request = image_request_from_job(
            {
                "payload": {
                    "projectId": "project-1",
                    "mode": "edit_image",
                    "model": model,
                    "prompt": "repaint the sky",
                    "sourceAssetId": "asset-source",
                    "width": 16,
                    "height": 16,
                    "count": 1,
                }
            }
        )
        result = adapter._run_pipeline(
            SimpleNamespace(gpu_id="cpu"),
            FakePipe(),
            request,
            7,
            project_path,
        )
        assert seen_paths == [project_path], f"{adapter.id} edit branch must call load_source_image(project_path, ...)"
        assert result is FakeOutput.images[0]

def test_diffusers_image_adapters_call_apply_sampler(tmp_path, monkeypatch):
    """Epic 1753 sc-1762: Z-Image and Qwen ``_run_pipeline`` must thread the
    sampler/scheduler/shift selection into ``apply_sampler`` so the user's
    choice actually swaps the pipeline scheduler. A regression here would
    silently ignore the advanced fields and leave the model on its default
    scheduler — exactly the bug class this epic exists to prevent.
    """

    class FakeImage:
        def convert(self, _mode):
            return self

    class FakeOutput:
        images = [FakeImage()]

    class FakePipe:
        scheduler = None

        def __call__(self, **kwargs):
            return FakeOutput()

    class FakeTorch:
        @staticmethod
        def Generator(_device):
            class Gen:
                def manual_seed(self, _seed):
                    return self

            return Gen()

    monkeypatch.setattr(
        "scene_worker.image_adapters.importlib.import_module",
        lambda name: FakeTorch if name == "torch" else importlib.import_module(name),
    )

    captured: list[tuple[str, ...]] = []

    def fake_apply_sampler(pipe, sampler, scheduler, shift, *, adapter=None):
        captured.append((adapter, sampler, scheduler, shift))
        return {"sampler": sampler, "scheduler": scheduler, "shift": shift, "noop": False}

    monkeypatch.setattr("scene_worker.image_adapters.apply_sampler", fake_apply_sampler)

    for adapter, model in (
        (ZImageDiffusersAdapter(), "z_image_turbo"),
        (QwenImageAdapter(), "qwen_image"),
    ):
        captured.clear()
        request = image_request_from_job(
            {
                "payload": {
                    "projectId": "project-1",
                    "mode": "text_to_image",
                    "model": model,
                    "prompt": "still life with sampler swap",
                    "width": 16,
                    "height": 16,
                    "count": 1,
                    "advanced": {
                        "sampler": "dpmpp",
                        "scheduler": "karras",
                    },
                }
            }
        )
        adapter._run_pipeline(
            SimpleNamespace(gpu_id="cpu"),
            FakePipe(),
            request,
            7,
            tmp_path,
        )
        assert captured == [(adapter.id, "dpmpp", "karras", None)], (
            f"{adapter.id} did not thread sampler/scheduler into apply_sampler"
        )

    # Shift axis: scheduler == "shift" must pass through a numeric shift value.
    captured.clear()
    request = image_request_from_job(
        {
            "payload": {
                "projectId": "project-1",
                "mode": "text_to_image",
                "model": "z_image_turbo",
                "prompt": "shift test",
                "width": 16,
                "height": 16,
                "count": 1,
                "advanced": {
                    "sampler": "euler",
                    "scheduler": "shift",
                    "schedulerShift": 4.5,
                },
            }
        }
    )
    ZImageDiffusersAdapter()._run_pipeline(
        SimpleNamespace(gpu_id="cpu"),
        FakePipe(),
        request,
        9,
        tmp_path,
    )
    assert captured == [("z_image_diffusers", "euler", "shift", pytest.approx(4.5))]

def test_character_image_capability_implies_engine_or_tuning_declaration():
    """Every builtin model that advertises `character_image` must have either
    a worker engine block (`ipAdapter` / `instantId` in MODEL_TARGETS) OR a
    `ui.variationStrength` declaration in the manifest. Otherwise the capability
    flag is dishonest — the picker shows the model in "With character" mode but
    the worker silently ignores the reference, the same shape as z_image_turbo's
    pre-sc-2005 bug. This is the cross-backbone guard for epic 2003 (sc-2018):
    adding a future character_image backbone without engine wiring will fail
    here before it ever reaches a user.
    """
    manifest = _load_builtin_models_manifest()
    misleading: list[str] = []
    for model in manifest.get("models", []):
        capabilities = model.get("capabilities") or []
        if "character_image" not in capabilities:
            continue
        target = MODEL_TARGETS.get(model["id"], {})
        ui = model.get("ui") or {}
        has_engine = bool(target.get("ipAdapter") or target.get("instantId") or target.get("pulidFlux"))
        has_variation_ui = bool(ui.get("variationStrength"))
        if not (has_engine or has_variation_ui):
            misleading.append(model["id"])
    assert not misleading, (
        f"Models advertise `character_image` without engine wiring or a "
        f"`ui.variationStrength` declaration: {misleading}. Add an `ipAdapter`, "
        f"`instantId`, or `pulidFlux` block in MODEL_TARGETS for an IP-Adapter / "
        f"face-ID backbone, or declare `ui.variationStrength` for an edit-style "
        f"backbone (sc-2017), or drop the capability flag (the z_image_turbo bug, "
        f"sc-2005)."
    )

def test_kolors_declares_strict_pose_controlnet():
    """sc-2264: Kolors is the strict pose tier — the manifest must advertise
    ui.poseLibrary AND the worker target must carry the controlNetPose config so
    the pose picker offers it and the adapter can load the pose ControlNet."""
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    kolors = manifest_by_id.get("kolors", {})
    assert kolors.get("ui", {}).get("poseLibrary") is True, (
        "kolors must declare ui.poseLibrary so the pose picker offers the strict tier (sc-2264)."
    )
    target = MODEL_TARGETS.get("kolors", {})
    assert target.get("controlNetPose", {}).get("repo") == "Kwai-Kolors/Kolors-ControlNet-Pose", (
        "kolors MODEL_TARGETS must carry the Kolors-ControlNet-Pose repo for the strict pose path."
    )
    # Identity still rides the IP-Adapter; the pose path composes both.
    assert target.get("ipAdapter"), "kolors pose path needs the IP-Adapter for identity."

def test_models_with_engine_block_advertise_character_image():
    """The reverse-drift guard. Any model that ships an `ipAdapter` or
    `instantId` block in MODEL_TARGETS exists to serve Character Studio's
    reference flow — the manifest must advertise the capability so the picker
    surfaces it. Catches the case where someone wires the worker engine but
    forgets to flip the manifest flag, leaving the engine unreachable.
    """
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    unreachable: list[str] = []
    for model_id, target in MODEL_TARGETS.items():
        if not (target.get("ipAdapter") or target.get("instantId") or target.get("pulidFlux")):
            continue
        builtin = manifest_by_id.get(model_id)
        if builtin is None:
            # Worker-only target not exposed as a built-in (unwired path).
            continue
        capabilities = builtin.get("capabilities") or []
        if "character_image" not in capabilities:
            unreachable.append(model_id)
    assert not unreachable, (
        f"Models have engine blocks in MODEL_TARGETS but the builtin manifest "
        f"does not advertise `character_image`: {unreachable}. Add the capability "
        f"to `capabilities` and `ui.recommendedFor` so the Image Studio "
        f"\"With character\" picker surfaces the model."
    )

def test_hide_reference_strength_models_declare_a_variation_knob():
    """Symmetry guard for the sc-2017 picker UX. A model that opts out of the
    IP-Adapter reference-strength slider via `ui.hideReferenceStrength` MUST
    also declare `ui.variationStrength` — otherwise the picker shows no tuning
    control at all, and the worker silently runs at default true_cfg_scale.
    """
    manifest = _load_builtin_models_manifest()
    unbalanced: list[str] = []
    for model in manifest.get("models", []):
        ui = model.get("ui") or {}
        if not ui.get("hideReferenceStrength"):
            continue
        if not ui.get("variationStrength"):
            unbalanced.append(model["id"])
    assert not unbalanced, (
        f"Models hide the Reference-strength slider without declaring "
        f"`ui.variationStrength`: {unbalanced}. The picker would leave the user "
        f"with NO identity tuning control. Add `variationStrength` or drop "
        f"`hideReferenceStrength`."
    )

def test_character_studio_angle_prompt_augments_cover_full_pack():
    """Every angle in the canonical ANGLE_SET_ORDER must have a matching
    prompt-augment string. Backbones that consume the shared list (Qwen, FLUX2,
    SenseNova) won't render correctly for an angle without an augment — the
    user would get an unaugmented prompt repeated for that index."""
    from scene_worker.character_studio_angles import (
        ANGLE_PROMPT_AUGMENTS,
        ANGLE_SET_ORDER,
        augment_prompt_for_angle,
    )

    missing = [angle for angle in ANGLE_SET_ORDER if not ANGLE_PROMPT_AUGMENTS.get(angle)]
    assert not missing, f"Angles missing prompt augments: {missing}"
    # Augment helper appends the per-angle clause to the user's base prompt.
    augmented = augment_prompt_for_angle("a portrait of the woman", "three_quarter_left")
    assert "three-quarter left profile" in augmented
    assert augmented.startswith("a portrait of the woman,")
    # Unknown angle is a no-op (caller passes the base through unchanged).
    assert augment_prompt_for_angle("hello", "unknown_angle_id") == "hello"

def test_prompt_driven_angle_backbones_share_the_instantid_angle_pack():
    """Each prompt-driven backbone (Qwen-Lightning, FLUX.2-klein, SenseNova) in
    the multi-backbone angle-set picker (sc-2003) MUST advertise the same 11
    canonical angles as the InstantID baseline. The picker depends on every
    backbone exposing the same angle ids so the dropdown shows identical
    angle options regardless of which backbone is selected.
    """
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    instantid_angles = (
        manifest_by_id.get("instantid_realvisxl", {}).get("ui", {}).get("viewAngles") or []
    )
    instantid_ids = {entry["id"] for entry in instantid_angles}
    assert len(instantid_ids) == 11, "Baseline angle pack drifted from 11 canonical angles."
    # The prompt-driven backbones we shipped in the picker matrix:
    for model_id in (
        "qwen_image_edit_2511_lightning",
        "flux2_klein_9b",
        "sensenova_u1_8b_fast",
    ):
        backbone = manifest_by_id.get(model_id) or {}
        view_angles = backbone.get("ui", {}).get("viewAngles") or []
        ids = {entry["id"] for entry in view_angles}
        assert ids == instantid_ids, (
            f"{model_id} advertises viewAngles that don't match InstantID's pack: "
            f"missing={instantid_ids - ids}, extra={ids - instantid_ids}. "
            f"The Character Studio angle dropdown depends on all four backbones "
            f"exposing the same 11 angle ids."
        )

def test_best_effort_backbones_declare_pose_library_capability():
    """sc-2256 / sc-2262: the best-effort pose backbones must advertise
    ui.poseLibrary so the Character Studio pose picker (poseModels filter)
    includes them alongside the strict InstantID backbone."""
    manifest = _load_builtin_models_manifest()
    manifest_by_id = {model["id"]: model for model in manifest.get("models", [])}
    instantid = manifest_by_id.get("instantid_realvisxl", {})
    assert instantid.get("ui", {}).get("poseLibrary") is True, "Baseline pose backbone drifted."
    for model_id, story in (
        ("qwen_image_edit_2511_lightning", "sc-2256"),
        ("flux2_klein_9b", "sc-2262"),
        # The other two klein variants share the same mlx_flux2 best-effort pose
        # path, so they're offered in the picker for output comparison too.
        ("flux2_klein_9b_kv", "sc-2262"),
        ("flux2_klein_9b_true_v2", "sc-2262"),
    ):
        entry = manifest_by_id.get(model_id, {})
        assert entry.get("ui", {}).get("poseLibrary") is True, (
            f"{model_id} must declare ui.poseLibrary so the pose picker offers its "
            f"best-effort tier ({story})."
        )

def test_qwen_image_adapter_angleset_loop_uses_augmented_prompts(monkeypatch):
    """When advanced.angleSet is set on a character_image request, the
    QwenImageAdapter loops the 11 canonical angles and routes each to
    _run_pipeline with the per-angle prompt_override — NOT the original
    user prompt. Sanity-check this without loading torch / diffusers."""
    import sys as _sys
    from types import ModuleType, SimpleNamespace as _SimpleNamespace

    from scene_worker.character_studio_angles import ANGLE_PROMPT_AUGMENTS, ANGLE_SET_ORDER
    from scene_worker.image_adapters import QwenImageAdapter

    # CI (sceneworks-core-pytest) installs requirements-dev.txt only — torch
    # isn't there. Stub the few attrs the generate() path touches so the test
    # runs against the pure-Python dispatch logic without pulling torch.
    if "torch" not in _sys.modules:
        torch_stub = ModuleType("torch")
        torch_stub.cuda = _SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        torch_stub.backends = _SimpleNamespace(mps=_SimpleNamespace(is_available=lambda: False))
        torch_stub.mps = _SimpleNamespace(empty_cache=lambda: None)
        monkeypatch.setitem(_sys.modules, "torch", torch_stub)

    captured: list[str] = []

    def fake_run_pipeline(
        self, settings, pipe, request, seed, project_path, *, cancel_requested=None, prompt_override=None, pose_skeleton=None
    ):
        from PIL import Image as _Image
        # Angle path carries NO skeleton — it's prompt-driven, not multi-image.
        assert pose_skeleton is None
        captured.append(prompt_override or request.prompt)
        return _Image.new("RGB", (8, 8))

    # Stub out the heavy load + LoRA paths.
    monkeypatch.setattr(QwenImageAdapter, "_run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(QwenImageAdapter, "_load_pipeline", lambda self, *args, **kwargs: object())
    monkeypatch.setattr(QwenImageAdapter, "_apply_loras", lambda self, *args, **kwargs: None)
    # Don't actually write asset files in the test — short-circuit the writer.
    from scene_worker import image_adapters as ia

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, **_kwargs):
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)
    # Bypass the GPU-backend-required check and the device-activation step:
    # both poke at torch internals the stub above doesn't fully model, and the
    # dispatch logic we're verifying doesn't depend on the answer.
    monkeypatch.setattr(ia, "require_inference_backend_for_gpu_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "activate_torch_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "select_torch_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *args, **kwargs: None)

    request = ia.ImageRequest(
        project_id="p",
        mode="character_image",
        prompt="a photo of the character",
        negative_prompt="",
        model="qwen_image_edit_2511_lightning",
        count=1,
        seed=42,
        seeds=[],
        width=1024,
        height=1024,
        style_preset="",
        loras=[],
        character_id=None,
        character_look_id=None,
        source_asset_id=None,
        reference_asset_id="ref-asset-id",
        advanced={"angleSet": True},
        model_manifest_entry={},
    )
    fake_settings = _SimpleNamespace(gpu_id="cpu")
    QwenImageAdapter().generate(
        settings=fake_settings,
        job={"id": "job_x", "payload": {}},
        request=request,
        project_path=None,
        progress=lambda *args, **kwargs: None,
        cancel_requested=lambda: False,
    )
    assert len(captured) == len(ANGLE_SET_ORDER) == 11
    # Each captured prompt should contain the corresponding angle's augment.
    for prompt, angle in zip(captured, ANGLE_SET_ORDER):
        augment = ANGLE_PROMPT_AUGMENTS[angle]
        assert augment in prompt, (
            f"angle '{angle}' didn't reach _run_pipeline as an augmented prompt: "
            f"expected snippet {augment!r} in {prompt!r}"
        )

def test_qwen_image_adapter_pose_loop_renders_skeleton_and_passes_multi_image(monkeypatch):
    """sc-2256: advanced.poses loops the selected library poses, renders each as
    an OpenPose skeleton, and routes it to _run_pipeline as pose_skeleton (so the
    reference branch builds image=[reference, skeleton]) with the pose prompt cue.
    Pose takes precedence over angleSet when both are present."""
    import sys as _sys
    from types import ModuleType, SimpleNamespace as _SimpleNamespace

    from scene_worker.character_studio_angles import POSE_SKELETON_PROMPT
    from scene_worker.image_adapters import QwenImageAdapter

    if "torch" not in _sys.modules:
        torch_stub = ModuleType("torch")
        torch_stub.cuda = _SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        torch_stub.backends = _SimpleNamespace(mps=_SimpleNamespace(is_available=lambda: False))
        torch_stub.mps = _SimpleNamespace(empty_cache=lambda: None)
        monkeypatch.setitem(_sys.modules, "torch", torch_stub)

    captured: list[dict] = []

    def fake_run_pipeline(
        self, settings, pipe, request, seed, project_path, *, cancel_requested=None, prompt_override=None, pose_skeleton=None
    ):
        from PIL import Image as _Image
        captured.append({"prompt_override": prompt_override, "pose_skeleton": pose_skeleton})
        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(QwenImageAdapter, "_run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(QwenImageAdapter, "_load_pipeline", lambda self, *args, **kwargs: object())
    monkeypatch.setattr(QwenImageAdapter, "_apply_loras", lambda self, *args, **kwargs: None)
    from scene_worker import image_adapters as ia

    class _FakeWriter:
        def write_incremental_outputs(self, *, image_count, image_at_index, **_kwargs):
            for index in range(image_count):
                image_at_index(index)
            return {"images": image_count}

    monkeypatch.setattr(ia, "ImageAssetWriter", _FakeWriter)
    monkeypatch.setattr(ia, "require_inference_backend_for_gpu_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "activate_torch_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(ia, "select_torch_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ia, "gpu_memory_snapshot", lambda *args, **kwargs: None)
    # draw_bodypose needs cv2 (not in the requirements-dev CI venv); the actual
    # render is covered by test_draw_bodypose_renders_colored. Here we only verify
    # dispatch, so stub it to a tiny array Image.fromarray can consume.
    import numpy as _np
    monkeypatch.setattr(ia, "draw_bodypose", lambda w, h, kps: _np.zeros((h, w, 3), dtype=_np.uint8))

    # Two library poses; flat 18-point COCO skeletons (values arbitrary but valid).
    kp = [[0.5, 0.1 + 0.04 * i] for i in range(18)]
    request = ia.ImageRequest(
        project_id="p",
        mode="character_image",
        prompt="a photo of the character",
        negative_prompt="",
        model="qwen_image_edit_2511_lightning",
        count=1,
        seed=42,
        seeds=[],
        width=64,
        height=64,
        style_preset="",
        loras=[],
        character_id=None,
        character_look_id=None,
        source_asset_id=None,
        reference_asset_id="ref-asset-id",
        # angleSet also set to prove pose takes precedence (no angle loop runs).
        advanced={"poses": [{"id": "sit_01", "keypoints": kp}, {"id": "stand_01", "keypoints": kp}], "angleSet": True},
        model_manifest_entry={},
    )
    QwenImageAdapter().generate(
        settings=_SimpleNamespace(gpu_id="cpu"),
        job={"id": "job_pose", "payload": {}},
        request=request,
        project_path=None,
        progress=lambda *args, **kwargs: None,
        cancel_requested=lambda: False,
    )
    # One image per pose (not the 11-angle loop), each with a rendered skeleton
    # and the pose cue appended to the prompt.
    assert len(captured) == 2
    for entry in captured:
        assert entry["pose_skeleton"] is not None
        assert POSE_SKELETON_PROMPT in (entry["prompt_override"] or "")

def test_load_state_dict_requires_weights_only(tmp_path):
    """sc-4230 / F-WORKER-6: .pth upscaler weights load with weights_only=True
    (no pickle execution), and there is NO silent fallback to an unrestricted
    torch.load — an ancient torch without the flag fails loudly instead."""
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"not really a checkpoint")

    class _SafeTorch:
        def load(self, path, map_location=None, weights_only=False):
            assert weights_only is True, "must load with weights_only=True"
            return {"module.body.weight": 1, "other": 2}

    state = _load_state_dict(_SafeTorch(), weights)
    # `module.` prefix stripped; the safe load result is returned.
    assert state == {"body.weight": 1, "other": 2}

    class _AncientTorch:
        """torch.load that predates weights_only — it must NOT be retried unsafely."""

        def load(self, path, map_location=None, **kwargs):
            if "weights_only" in kwargs:
                raise TypeError("load() got an unexpected keyword argument 'weights_only'")
            raise AssertionError("must not fall back to an unrestricted torch.load")

    with pytest.raises(RuntimeError, match="weights_only"):
        _load_state_dict(_AncientTorch(), weights)
