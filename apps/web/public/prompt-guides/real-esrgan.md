# Real-ESRGAN Upscaling Guide

Real-ESRGAN is a post-process upscaler. It runs after an image has already been generated, so the prompt, seed, LoRAs, and base model determine the source image before this step begins.

## When to Enable

Use 2x when you want a larger working image with minimal texture change.

Use 4x when the source image is clean and you need a much larger export or crop target.

Keep upscaling off for quick drafts, prompt iteration, or images that already have visible artifacts. Upscaling can make existing artifacts more obvious.

## Practical Notes

The original image is retained alongside the upscaled variant.

Upscaling uses pixel-space weights and does not rerun the image model.

Large 4x outputs use more memory. If a job fails on a large image, retry with 2x or a smaller source resolution.
