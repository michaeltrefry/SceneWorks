# Qwen Image Edit (2511) Prompt Guide

## Best For

Character Studio reference generation, localized image editing, and subject-consistency tasks where you want the **same person in a new scene, pose, or style** while preserving identity. The December 2025 release of Qwen-Image-Edit (Qwen/Qwen-Image-Edit-2511) is the third iteration in the family and the current default for Qwen-based edit and character work. Apache-2.0, ungated.

For text-to-image only (no reference), use **Qwen Image**. For ~10× faster inference at a small quality cost, use **Qwen Image Edit (2511) Lightning**.

## What's New In 2511

Building on the September 2509 release, 2511 brings:

- **Mitigated image drift** — repeated edits and high-step runs stay closer to the source.
- **Improved character consistency in multi-person scenes** — 2509 was strong on single subjects; 2511 holds identity for groups too.
- **Integrated popular LoRAs** — common style/quality LoRAs are baked into the base, so you get their effects without loading them separately.
- **Stronger geometric reasoning** — better at constraints like camera angle, perspective, and object placement.

The pipeline shape is unchanged from 2509: same `QwenImageEditPlusPipeline`, same `image=` + `true_cfg_scale` knobs.

## How It Works

Qwen-Image-Edit doesn't blend a reference embedding into the diffusion process the way IP-Adapter models do. The reference image goes through **two parallel encoders**:

- **Qwen2.5-VL** for visual *semantics* (who the subject is, what the scene contains)
- **VAE encoder** for visual *appearance* (color, texture, lighting)

The diffusion then renders the prompt while both encoders steer toward the reference. The variation knob is `trueCfgScale`:

- **High `trueCfgScale` (5–6)** = prompt-dominant → more variation from the reference (new pose, new outfit, new scene work better here).
- **Low `trueCfgScale` (~1–2)** = reference-dominant → closer to the source (subtle prompt-driven adjustments, style transfer).
- **Default 4.0** is the model-card sweet spot.

`guidanceScale` should be left at 1.0 (the 2511 recipe disables classifier-free guidance on the conditioning side; `true_cfg_scale` carries the steering).

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

The model preserves everything you don't mention, and 2511's drift mitigation makes this more reliable than 2509 on multi-pass edits.

## Tips

- **For Character Studio**: lead with "the same character/person/subject" so the model treats the reference as identity rather than as a scene to modify. Avoid "of the woman in the reference" — that often reproduces the reference's composition.
- **Negative prompts** are required for `trueCfgScale > 1` to function. Even an empty string is accepted; common defaults: `lowres, deformed, oversaturated, distorted face, watermark`.
- **Resolution**: 1024×1024 is the trained center; canonical aspect-ratio buckets (768×768, 1280×720, 720×1280) work well.
- **Multi-person scenes**: 2511 holds group identity better than 2509 — name each subject in the prompt ("the same two characters, walking side by side …") to let the model track them.
- **Multi-image references**: supply multiple approved references for stronger identity averaging. Useful for invented characters with multiple hero shots.
- **Step count**: 40 is the 2511 recipe (vs. 50 for 2509). Drop to 30 for speed if quality holds.

## Comparison To Other Character Studio Backbones

| Backbone | Identity tier | When to pick |
|---|---|---|
| **InstantID (RealVisXL)** | Faithful face geometry (ArcFace + landmarks) | Highest-fidelity face likeness for real people |
| **Kolors / SDXL / RealVisXL IP-Adapter** | Resemblance (CLIP/face embed) | Scene-flexible "looks like" without faithful identity |
| **FLUX IP-Adapter** | Resemblance (XLabs CLIP-L) | Scene-flexible resemblance on FLUX's quality |
| **Qwen Image Edit (2511)** | **Semantic + appearance** of the whole reference | Subject + outfit + setting continuity, varied poses/scenes, multi-person, multi-reference |

Qwen complements the IP-Adapter family — it carries more of the reference's *context* (outfit, lighting) along with the subject, where IP-Adapter focuses on the subject alone.

## Sources

- [Qwen-Image-Edit-2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Qwen-Image-Edit-2511 announcement](https://qwen.ai/blog?id=qwen-image-edit-2511)
- [Diffusers QwenImage pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage)
