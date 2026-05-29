# Qwen Image Edit (2511) Lightning Prompt Guide

## Best For

Fast Character Studio iteration and quick image edits on top of Qwen-Image-Edit-2511 — about **10× faster** than the base 40-step model. Same Qwen2.5-VL + VAE dual-control identity behavior as the full 2511 release; trades a small amount of quality for speed.

Use **Qwen Image Edit (2511)** when you want the full-quality 40-step result; use Lightning for iteration, batch passes, and exploratory work where wall-clock dominates.

## How It Works

The lightx2v 4-step Lightning LoRA is fused into the Qwen-Image-Edit-2511 base on first load and stays cached in-process. The fused model runs in **4 inference steps** with classifier-free guidance disabled (`guidanceScale=1.0`, `trueCfgScale=1.0`) per the distill recipe.

User LoRAs (style, character, enhancement) still apply on top of the fused base via the normal LoRA slot — the Lightning LoRA is absorbed into the base weights, leaving the slot free.

## Prompt Shape

Same as the base 2511 model — the prompting recipe doesn't change, only the step count and CFG.

For Character Studio:

`The same character + new context + scene/lighting/composition + style anchor.`

For localized edits:

`Describe the modification only; the model preserves the rest.`

See the **[Qwen Image Edit (2511) Prompt Guide](qwen-image-edit-2511.md)** for full prompting structure and Character Studio examples.

## Tips

- **Don't override step count**: 4 is the trained sweet spot. Higher step counts on a distilled model degrade quality (overshooting the schedule).
- **Don't override `trueCfgScale`**: Lightning is distilled at cfg=1.0; raising it produces ghosting and color shift. The variation slider is intentionally hidden in this model's UI.
- **Use for ideation, switch for finals**: prototype prompts and references with Lightning, then re-render the keepers with **Qwen Image Edit (2511)** for max quality.
- **First run downloads ~1 GB** for the Lightning LoRA from `lightx2v/Qwen-Image-Edit-2511-Lightning`; subsequent runs hit the cache.
- **User LoRAs still work**: character/style LoRAs you've trained or imported apply normally on top of the fused base.

## Sources

- [Qwen-Image-Edit-2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [lightx2v Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
- [Qwen-Image-Lightning GitHub (recipe)](https://github.com/ModelTC/Qwen-Image-Lightning)
