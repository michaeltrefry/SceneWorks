# SceneWorks Worker

Python worker package for Diffusers/PyTorch-backed image and video inference
adapters.

In Docker Compose this worker handles Diffusers/PyTorch inference and the
Rust worker handles utility jobs, including person detection and tracking.
Python utility fallbacks are explicit and limited to legacy rollback flags:

```text
SCENEWORKS_PYTHON_UTILITY_JOBS=1
```

- `SCENEWORKS_LEGACY_MODEL_LORA_JOBS=1`: `model_download`, `lora_import`
- `SCENEWORKS_LEGACY_FFMPEG_JOBS=1`: `frame_extract`, `timeline_export`
