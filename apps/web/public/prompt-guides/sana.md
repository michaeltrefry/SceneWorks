# SANA Prompt Guide

## Best For

Fast, efficient 1024×1024 text-to-image. SANA 1600M is NVIDIA's Linear-DiT model — a 1.6B linear-attention transformer paired with a deep-compression 32× DC-AE autoencoder and a Gemma-2 instruction text encoder. It uses **real classifier-free guidance** (positive *and* negative prompt) and renders quickly with a comfortably small memory footprint.

Distributed under the NVIDIA Open Model License (non-commercial / research use only).

## Prompt Shape

SANA's text encoder is the Gemma-2 instruction model run over a Complex-Human-Instruction (CHI) wrapper, so it responds well to fluent, natural-language descriptions rather than tag lists. Write a sentence or two describing the finished image:

`subject + setting + visual details + style + composition + lighting`

Use the positive prompt to describe what you want and the negative prompt to push away what you don't. Recommended defaults: **~20 steps at guidance 4.5**.

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo — type, action, position, distinguishing features.

Good: `a red fox curled asleep on a mossy log in a misty forest`

### Setting And Details

Add the environment, materials, and any small details you want emphasized. SANA follows descriptive language closely.

### Style And Lighting

Name the medium (photo, oil painting, 3D render), the mood, and the lighting (golden hour, soft studio light, dramatic rim light).

## Settings

- **Steps**: ~20 (the diffusers reference default). More steps add little once the image converges.
- **Guidance**: ~4.5. Lower for more natural/varied results, higher for stricter prompt adherence.
- **Resolution**: native 1024×1024; width and height must be multiples of 32 (the DC-AE compression factor).
- **Negative prompt**: supported — use it to remove unwanted elements or artifacts.

## Notes

SANA ships dense (bf16); there is no quantized variant. Generations are for non-commercial use only.
