# Qwen Image Edit (2509) Prompt Guide

## Best For

Character Studio reference generation, localized image editing, and subject-consistency tasks where you want the **same person in a new scene, pose, or style** while preserving identity. The September 2509 iteration of Qwen-Image-Edit is built around the model card's headline use case: *"changing a person's pose while maintaining excellent identity consistency."* Apache-2.0, ungated.

For text-to-image only (no reference), use **Qwen Image**. For straightforward single-edit jobs without subject preservation, the earlier **Qwen Image Edit** (August) is interchangeable.

## How It Works

Unlike IP-Adapter models (Kolors, SDXL, FLUX), Qwen-Image-Edit doesn't blend a reference embedding into the diffusion process — it feeds the reference image into **two parallel encoders**:

- **Qwen2.5-VL** for visual *semantics* (who the subject is, what the scene contains)
- **VAE encoder** for visual *appearance* (color, texture, lighting)

The diffusion then renders the prompt while both encoders steer toward the reference. The variation knob is `trueCfgScale` rather than a reference-strength slider:

- **High `trueCfgScale` (5–6)** = prompt-dominant → more variation from the reference (new pose, new outfit, new scene work better here).
- **Low `trueCfgScale` (~1–2)** = reference-dominant → closer to the source (subtle prompt-driven adjustments, style transfer).
- **Default 4.0** is the model-card sweet spot.

## Prompt Shape For Character Studio

When you pick a character with an approved reference, the reference becomes the model's `image=` input — your prompt describes **what the same character is doing now**, not the reference itself.

Effective structure:

`same subject + new context + scene/lighting/composition + style anchor`

### Examples

`The same character at a sunlit beach café, reading a paperback, soft morning haze, candid editorial photograph, shallow depth of field.`

`The same person on a foggy New York street at night, wearing a long wool coat, neon storefront reflections, cinematic 35mm.`

`The same character in a sunlit kitchen, mid-laugh, holding a coffee mug, documentary photograph, warm window light.`

## Prompt Shape For Edit Mode

When using the model for localized edits (no character reference, just a source image), describe the **modification**, not the whole scene:

- `Remove the watermark in the bottom-right corner.`
- `Change the background to a snowy mountain at dusk; keep the subject and pose unchanged.`
- `Replace the green shirt with a navy turtleneck.`

The model preserves everything you don't mention.

## Tips

- **For Character Studio**: lead with "the same character/person/subject" so the model treats the reference as identity rather than as a scene to modify. Avoid "of the woman in the reference" — that often reproduces the reference's composition.
- **Negative prompts** are required for `trueCfgScale > 1` to function. Even an empty string is accepted; common defaults: `lowres, deformed, oversaturated, distorted face, watermark`.
- **Resolution**: 1024×1024 is the trained center; canonical aspect-ratio buckets (768×768, 1280×720, 720×1280) work well.
- **Don't fight the dual-control architecture**: long lists of "high quality, masterpiece, 8k" tags don't help much — the semantic+appearance encoders are doing that work from the reference.
- **Multi-image references** (Edit Plus pipeline only): supply multiple approved references for stronger identity averaging. Useful for invented characters with multiple hero shots.
- **trueCfgScale sweep**: try 2 / 4 / 6 in early sessions to find the right variation amount per character. Photoreal characters often want ~4; stylized/painted characters often want ~3.

## Comparison To Other Character Studio Backbones

| Backbone | Identity tier | When to pick |
|---|---|---|
| **InstantID (RealVisXL)** | Faithful face geometry (ArcFace + landmarks) | Highest-fidelity face likeness for real people |
| **Kolors / SDXL / RealVisXL IP-Adapter** | Resemblance (CLIP/face embed) | Scene-flexible "looks like" without faithful identity |
| **FLUX IP-Adapter** | Resemblance (XLabs CLIP-L) | Scene-flexible resemblance on FLUX's quality |
| **Qwen Image Edit (2509)** | **Semantic + appearance** of the whole reference | Subject + outfit + setting continuity, varied poses/scenes, multi-reference |

Qwen complements the IP-Adapter family — it carries more of the reference's *context* (outfit, lighting) along with the subject, where IP-Adapter focuses on the subject alone.

## Sources

- [Qwen-Image-Edit-2509 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [Qwen-Image-Edit model card](https://huggingface.co/Qwen/Qwen-Image-Edit) (August iteration)
- [Diffusers QwenImage pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)
