"""Vendored Kolors ControlNet (sc-2264).

Source: github.com/Kwai-Kolors/Kolors @ master, Apache-2.0.
  models/controlnet.py  -> models/controlnet.py (verbatim; self-contained, diffusers-only)
  pipelines/pipeline_controlnet_xl_kolors_img2img.py -> verbatim
    (only internal import: `from ..models.controlnet import ControlNetModel`)

Why vendored: diffusers has no native KolorsControlNetPipeline (upstream issues
8801/9315 open). The pipeline uses diffusers UNet2DConditionModel, so it composes
with the same Kolors components diffusers.KolorsPipeline loads. Used to add the
strict pose tier (Kwai-Kolors/Kolors-ControlNet-Pose) + IP-Adapter-Plus identity.
"""
