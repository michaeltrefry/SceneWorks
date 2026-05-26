# AuraSR Upscaling Guide

AuraSR v2 is a 4x post-process upscaler for generated and real-world images. It runs after the base image is saved, so prompts, seeds, LoRAs, and the selected image model determine the source image before AuraSR adds detail.

Use AuraSR when you want a higher-quality 4x upscale and can spend more memory and time than Real-ESRGAN. The original image is retained alongside the upscaled variant.

AuraSR v2 uses Apache-2.0 weights from `fal/AuraSR-v2`. The older `fal/AuraSR` checkpoint uses CC-BY-SA-4.0 and is not wired into SceneWorks.
