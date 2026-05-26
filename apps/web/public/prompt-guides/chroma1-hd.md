# Chroma1-HD Prompt Guide

## Best For

High-resolution, detail-rich text-to-image: photography, illustration, concept art, and editorial work. Chroma1-HD is the high-resolution tune of the Chroma1 family — an 8.9B model derived from FLUX.1-schnell, Apache-2.0 and commercial-safe. Unlike distilled FLUX, Chroma uses **real classifier-free guidance**, so you get both a positive *and* a negative prompt.

## Choosing A Chroma1 Variant

- **Chroma1-HD** — the default. Best fidelity and the high-resolution sweet spot. Use it for finished images.
- **Chroma1-Base** — the neutral foundation the family is finetuned from. Same engine and prompt shape as HD; pick it when you want a clean base (e.g. for LoRA training) or a slightly less HD-biased look.
- **Chroma1-Flash** — the fast, CFG-baked variant (~8 steps, no negative prompt). Use it for quick drafts and iteration.

HD, Base, and Flash share one worker adapter and the same prompt structure — only step count and guidance differ.

## Prompt Shape

Chroma was trained on rich natural-language captions (T5-XXL text encoder, no CLIP), so write a fluent sentence or two describing the finished image rather than a list of tags:

`subject + setting + visual details + style + composition + lighting + any text`

Because HD uses real CFG, write the positive prompt as a description of what you want and use the negative prompt to push away what you don't. Recommended defaults: **~40 steps at guidance 3.0**.

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo — type, action, position, distinguishing features.

Good: `a weathered fisherman mending a net on a wooden dock at dawn`

### Details

Add material, texture, and atmosphere:

- `hand-stitched leather`
- `condensation on cold glass`
- `drifting morning fog`
- `intricate filigree pattern`

### Style

- `editorial fashion photography`
- `cinematic film still, 35mm`
- `flat vector illustration`
- `soft gouache painting`

### Camera And Composition

- `low-angle hero shot`
- `extreme close-up macro`
- `wide establishing shot`
- `centered product shot, studio backdrop`

### Lighting

- `golden hour backlight`
- `soft window light`
- `neon-lit night scene`
- `dramatic rim lighting`

### Text In Images

Chroma renders legible text — quote the exact words and describe the medium:

`a vintage enamel sign reading "HARBOR CAFE" in cream serif letters`

## Negative Prompts

HD honors a negative prompt (guidance > 1). Keep it short and targeted:

`blurry, lowres, deformed, extra fingers, watermark, oversaturated, flat colors`

## Tips

- ~40 steps at guidance 3.0 is the sweet spot; raise guidance for stronger prompt adherence, lower it for more natural variation.
- Use the negative prompt — it is active on HD/Base, unlike distilled models.
- 1024×1024 is the native sweet spot; portrait/landscape buckets work well.
- For fast drafts, switch to Chroma1-Flash; for a finetuning base, use Chroma1-Base.

## Example Prompts

`A cozy independent bookstore storefront at dusk, warm interior glow spilling onto a rain-slick cobblestone street, a hand-painted sign reading "PAGE & QUILL" in gold script, reflections in the wet pavement, cinematic shallow depth of field.`

`A studio product shot of a matte-black ceramic pour-over coffee set on a pale oak table, soft diffused side light, subtle steam rising, minimalist composition, high detail.`

## Sources

- [Chroma1-HD model card](https://huggingface.co/lodestones/Chroma1-HD)
- [Chroma1-Base model card](https://huggingface.co/lodestones/Chroma1-Base)
- [Diffusers Chroma pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/chroma)
