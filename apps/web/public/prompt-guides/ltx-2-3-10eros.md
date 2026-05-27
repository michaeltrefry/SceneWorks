# LTX-2.3 10Eros Prompt Guide

10Eros is a community LTX-2.3 merge tuned for image-to-video. It uses the same prompt shape as LTX-2.3, but is more literal: it has low self-reasoning, so motion, scene evolution, and audio must be commanded explicitly. Vague prompts produce static or drifting clips.

## Best For

Image-to-video and text-to-video short clips where you can describe the action precisely. Start from a strong source image for I2V.

## Prompt Shape

Write a single flowing paragraph:

`shot + scene + action over time + character details + camera movement + lighting/atmosphere + audio or dialogue`

Aim for 4 to 8 present-tense sentences. Spell out every beat of motion you want — the model will not infer it.

## Command The Motion

- State what moves, in what order, and where it ends: `she turns her head left, then looks down, then smiles`.
- Give camera moves a destination: `the camera slowly pushes in to a close-up of her face`.
- Name lighting and atmosphere directly: `warm key light from the left, soft shadows, shallow depth of field`.
- For audio, put spoken words in quotes and add delivery cues (language, tone, volume).

## Distill LoRAs

10Eros pairs with TenStrip's distilled LoRA experiments — use the `cond_safe` versions. The model card warns that larger distill LoRAs harm the fine-tune, so prefer the lighter cond_safe variant. Any LTX-video LoRA is compatible (same `ltx-video` family).

## What To Avoid

Avoid overloaded scenes, many simultaneous actions, conflicting lighting, and relying on readable text or logos. Keep physics simple.

## Sources

- [LTX official prompting guide](https://docs.ltx.video/api-documentation/guides/prompting-guide)
- [TenStrip LTX2.3-10Eros model card](https://huggingface.co/TenStrip/LTX2.3-10Eros)
