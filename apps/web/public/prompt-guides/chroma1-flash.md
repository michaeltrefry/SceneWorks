# Chroma1-Flash Prompt Guide

## Best For

Fast drafts and rapid iteration. Chroma1-Flash is the CFG-baked, low-step variant of the Chroma1 family — an 8.9B FLUX.1-schnell-derived model, Apache-2.0 and commercial-safe. It trades the negative prompt for speed: at the baked guidance of 1.0, classifier-free guidance is off, so put everything you want into the positive prompt.

## Choosing A Chroma1 Variant

- **Chroma1-Flash** — fastest (~8 steps, CFG baked, **no negative prompt**). Use it for quick drafts and iteration.
- **Chroma1-HD** — the high-resolution tune with real CFG + negative prompt (~40 steps). Use it for finished images.
- **Chroma1-Base** — the neutral foundation (real CFG, ~40 steps); the recommended base for finetuning.

All three share one worker adapter and the same prompt structure — only step count and guidance differ. A draft you like on Flash will compose similarly on HD at higher fidelity.

## Prompt Shape

Chroma was trained on rich natural-language captions (T5-XXL text encoder, no CLIP), so write a fluent sentence or two describing the finished image rather than a list of tags:

`subject + setting + visual details + style + composition + lighting + any text`

Flash bakes CFG (guidance 1.0), so it does **not** use a negative prompt — put everything you want into the positive prompt. Recommended default: **~8 steps** with a Heun-friendly schedule.

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo — type, action, position, distinguishing features.

Good: `a red fox curled asleep on a mossy log in a misty forest`

### Details

Add material, texture, and atmosphere:

- `dew on spiderwebs`
- `worn brass fittings`
- `soft volumetric light`
- `weathered concrete`

### Style

- `editorial photography`
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

`a chalkboard reading "TODAY'S SPECIAL" in looping white script`

## Tips

- Keep the prompt descriptive and positive; at guidance 1.0 there is no negative prompt.
- ~8 steps is the sweet spot — more steps rarely help Flash. Heun / DPM++ SDE samplers do well at low step counts.
- Use Flash to explore composition quickly, then re-render the keeper on Chroma1-HD for maximum quality.

## Example Prompts

`A cozy independent bookstore storefront at dusk, warm interior glow spilling onto a rain-slick cobblestone street, a hand-painted sign reading "PAGE & QUILL" in gold script, cinematic shallow depth of field.`

`A studio product shot of a matte-black ceramic pour-over coffee set on a pale oak table, soft diffused side light, subtle steam rising, minimalist composition.`

## Sources

- [Chroma1-Flash model card](https://huggingface.co/lodestones/Chroma1-Flash)
- [Chroma1-HD model card](https://huggingface.co/lodestones/Chroma1-HD)
- [Diffusers Chroma pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/chroma)
