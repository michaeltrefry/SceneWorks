# SceneWorks Worker

Python worker package for Diffusers/PyTorch-backed image and video inference
adapters.

In Docker Compose this worker handles Diffusers/PyTorch inference and the
procedural person-tracking fallback by default:

```text
SCENEWORKS_PYTHON_UTILITY_JOBS=1
```

Rust utility workers claim model downloads, LoRA imports, frame extraction, and
timeline exports by default. Python utility fallbacks are explicit:

- `SCENEWORKS_PYTHON_UTILITY_JOBS=1`: `person_detect`, `person_track`
- `SCENEWORKS_LEGACY_MODEL_LORA_JOBS=1`: `model_download`, `lora_import`
- `SCENEWORKS_LEGACY_FFMPEG_JOBS=1`: `frame_extract`, `timeline_export`
