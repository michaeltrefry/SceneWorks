# FLUX.2 [klein] 9B Prompt Guide

## Best For

A 9B-parameter Black Forest Labs FLUX.2 [klein] checkpoint, 4-step distilled — fast, high-quality text-to-image AND reference-driven editing in a single model. Apple Silicon only (MLX backend); ships in two variants:

- **FLUX.2 [klein] 9B** — text-to-image and reference editing.
- **FLUX.2 [klein] 9B-KV** — reference editing with KV-cache acceleration, ~2.4× faster than the base 9B edit path. Reference image required (cache is meaningless without one).

> **License:** FLUX Non-Commercial License — gated Hugging Face download. Accept the license at the model card and add a Hugging Face token under Settings → Service credentials before downloading. Generations are for non-commercial use only.

> **Hardware:** ~36 GB bf16 peak at 1024² on M5 Max. Q8 quantization (default) trades minor quality for ~25 % memory reduction. macOS only.

## Prompt Shape

FLUX.2 uses an 8B Qwen3 text embedder, so write a fluent natural-language description rather than tag lists:

`subject + setting + visual details + style + composition + lighting + any text`

There is no negative prompt — FLUX.2 disallows it. Keep everything in the positive prompt and describe what you DO want.

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo. Include type, action, position, and distinguishing features.

Good: `a marine biologist examining a glowing jellyfish in a research lab tank`

### Details

Add material, texture, and atmosphere:

- `brushed copper helmet`
- `wet kelp glinting under fluorescents`
- `late-afternoon golden light through high windows`

### Style + Composition

Pick concrete style cues (photo, illustration, render) and a composition (medium shot, low angle, etc.). The model handles photoreal and stylized equally well, but doesn't infer them — say what you want.

### Text

FLUX.2 [klein] renders short legible text well. Wrap exact strings in double quotes:

`a vintage tea-tin labeled "Earl Grey No. 3"`

## Reference Editing

When you attach a reference image, the model switches to `Flux2KleinEdit` and conditions on the reference at every step (or on cached reference KV for the -kv variant). Write the prompt as a description of the *target* image, not as an instruction:

Good: `a portrait of a black cat wearing a tall purple wizard hat with gold stars, sitting on a stack of vintage books, photoreal`

Avoid: `add a wizard hat to the cat`

The 4-step distillation means the cache-on -kv variant produces results in ~13 s on M5 Max, ~2.4× faster than the base 9B at the same prompt and reference.

## Defaults

- Resolution: 1024×1024 (also 768×768, 1280×720, 720×1280)
- Steps: 4 (the distillation target; longer runs don't help and aren't supported on -kv)
- Guidance: 1.0 (must stay at 1.0 on distilled klein variants)
- Quantization: Q8 (sweet spot — bf16 speed, lower memory)

## Sources

- [FLUX.2 [klein] 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- [FLUX.2 [klein] 9B-KV model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv)
- [FLUX.2 blog post](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)
