# RealVisXL Prompt Guide

## Best For

Photoreal text-to-image, image-to-image, and reference-based generation across portraits, products, landscapes, and editorial-style scenes. RealVisXL_V5.0 is a community photoreal SDXL finetune — it shares the SDXL architecture, sdxl-family LoRA support, real CFG + negative prompt, and the 1024×1024 native resolution, but pushes skin, light, and material rendering toward a less "shiny/plastic" look than base SDXL. openrail++, commercial use OK, ungated.

For faithful face identity from a reference image, use **InstantID (RealVisXL)** instead — same checkpoint, dedicated identity engine.

## Prompt Shape

RealVisXL responds best to **photoreal-leaning natural language** with concise quality tags. A reliable structure:

`subject + key details + style/medium + composition + lighting + quality tags`

CLIP encoders weight earlier tokens more heavily, so **lead with the subject and the most important attributes**.

## Build The Prompt

### Subject

Front-load the subject in plain language:

`a candid portrait of a woman in her early 30s holding a ceramic mug`

### Details

Add specific material, texture, and atmosphere — these are where RealVisXL pulls ahead of base SDXL:

- `fine skin texture, visible pores, light freckles`
- `soft cotton sweater, hand-knit cable pattern`
- `morning steam rising from the mug`
- `condensation beading on cold glass`

### Style / Medium

Reach for photographic vocabulary rather than illustrative terms:

- `editorial portrait photography, 50mm`
- `cinematic film still, 35mm anamorphic`
- `documentary travel photography`
- `studio product photography, matte backdrop`

### Camera And Composition

- `medium close-up, eye-level`
- `shallow depth of field, f/1.8`
- `low-angle hero shot`
- `rule of thirds, centered subject`

### Lighting

Photoreal output lives or dies on lighting language — be specific:

- `golden hour backlight, warm rim`
- `soft north-window light, even diffusion`
- `single key light, deep shadow falloff`
- `overcast daylight, neutral white balance`

### Quality Tags

Keep these short and photo-realistic — avoid stacking long lists of "8k, masterpiece, best quality" that flatten the result:

`sharp focus, natural skin tones, photorealistic, high detail`

## Negative Prompts

RealVisXL honors a negative prompt (guidance > 1). Use it to push away the common photoreal failure modes — keep it short and targeted:

`blurry, lowres, overprocessed skin, plastic skin, oversaturated, deformed, extra fingers, watermark, text, jpeg artifacts, painting, illustration`

## Tips

- ~30 steps at guidance 7.0 is a solid baseline; lower guidance (4–6) often yields more natural, less "baked" results — try both.
- Native 1024×1024; the canonical SDXL buckets (1152×896, 896×1152, 1216×832, 832×1216, 1344×768, 768×1344) are trained resolutions — prefer them.
- Lighting words do more work than style words — invest in specific, physically-plausible lighting.
- Keep skin/texture tags subtle; over-tagging "ultra-realistic skin" can produce the opposite (waxy or over-rendered) look.
- Layer sdxl-family LoRAs for specific styles or characters — RealVisXL accepts the entire SDXL LoRA ecosystem.
- With a character reference, use the reference-strength slider to balance prompt vs. likeness; faithful identity is **InstantID**'s job, not IP-Adapter's.

## Example Prompts

`A candid portrait of a fisherman in his sixties on a wooden dock at dawn, weathered hands holding a coil of rope, fine skin detail, soft golden backlight, shallow depth of field, editorial documentary photography, sharp focus.`

`Studio product shot of a brushed aluminum espresso tamper on warm walnut, soft directional side light, subtle wood grain, minimalist composition, professional product photography, natural color, high detail.`

## Sources

- [RealVisXL_V5.0 model card](https://huggingface.co/SG161222/RealVisXL_V5.0)
- [SDXL technical report](https://arxiv.org/abs/2307.01952)
- [Diffusers SDXL pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl)
