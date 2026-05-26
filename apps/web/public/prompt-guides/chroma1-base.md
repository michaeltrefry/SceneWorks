# Chroma1-Base Prompt Guide

## Best For

A neutral, general-purpose text-to-image foundation and the recommended starting point for **finetuning and LoRA training**. Chroma1-Base is the 8.9B FLUX.1-schnell-derived base the rest of the Chroma1 family is built from — Apache-2.0 and commercial-safe. Like HD it uses **real classifier-free guidance** (positive *and* negative prompt).

## Choosing A Chroma1 Variant

- **Chroma1-Base** — the neutral foundation. Pick it when you want a clean, un-opinionated base (e.g. as a LoRA training base) or general text-to-image without HD's high-resolution bias.
- **Chroma1-HD** — the high-resolution tune. Best fidelity for finished images; the default for most generation.
- **Chroma1-Flash** — the fast, CFG-baked variant (~8 steps, no negative prompt) for quick drafts.

All three share one worker adapter and the same prompt structure — only step count and guidance differ.

## Prompt Shape

Chroma was trained on rich natural-language captions (T5-XXL text encoder, no CLIP), so write a fluent sentence or two describing the finished image rather than a list of tags:

`subject + setting + visual details + style + composition + lighting + any text`

Base uses real CFG: write the positive prompt as a description of what you want and use the negative prompt to push away what you don't. Recommended defaults: **~40 steps at guidance 3.0**.

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo — type, action, position, distinguishing features.

Good: `a young chef plating a dessert in a bright modern kitchen`

### Details

Add material, texture, and atmosphere:

- `glossy chocolate ganache`
- `steam rising from fresh espresso`
- `soft natural window light`
- `intricate porcelain pattern`

### Style

- `editorial food photography`
- `cinematic film still, 35mm`
- `flat vector illustration`
- `traditional ink wash painting`

### Camera And Composition

- `low-angle hero shot`
- `extreme close-up macro`
- `wide establishing shot`
- `centered product shot, studio backdrop`

### Lighting

- `golden hour backlight`
- `soft diffused studio light`
- `neon-lit night scene`
- `dramatic rim lighting`

### Text In Images

Chroma renders legible text — quote the exact words and describe the medium:

`a wooden cafe sign reading "MORNING LIGHT" in warm hand-painted brush lettering`

## Negative Prompts

Base honors a negative prompt (guidance > 1). Keep it short and targeted:

`blurry, lowres, deformed, extra fingers, watermark, oversaturated, flat colors`

## Tips

- ~40 steps at guidance 3.0 is the sweet spot; adjust guidance to trade prompt adherence against natural variation.
- Use the negative prompt — it is active on Base/HD, unlike distilled models.
- As a finetuning base, Base is intentionally neutral — expect a less stylized look than HD out of the box.
- For finished high-resolution images use Chroma1-HD; for fast drafts use Chroma1-Flash.

## Example Prompts

`A sunlit artist's studio with canvases leaning against brick walls, paint-splattered wooden floor, a large north-facing window, dust motes drifting in the light, photographic, shallow depth of field.`

`A flat vector illustration of a mountain trail map, muted earth tones, clean line work, small labeled landmarks, balanced composition.`

## Sources

- [Chroma1-Base model card](https://huggingface.co/lodestones/Chroma1-Base)
- [Chroma1-HD model card](https://huggingface.co/lodestones/Chroma1-HD)
- [Diffusers Chroma pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/chroma)
