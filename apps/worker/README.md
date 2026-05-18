# SceneWorks Worker

Python worker package for Diffusers/PyTorch-backed image and video inference
adapters.

In Docker Compose this worker handles Diffusers/PyTorch inference and the
Rust worker handles utility jobs, including person detection and tracking.
The Python worker no longer advertises or runs utility jobs such as
`model_download`, `lora_import`, `frame_extract`, or `timeline_export`; keep the
Rust worker running alongside the API for those queues.
