# Krea 2 Prompt Guide

## Best For

Photorealistic and aesthetically-strong **still images** from natural-language prompts. Krea 2 is Krea
AI's from-scratch foundation image model — a single-stream rectified-flow DiT paired with a Qwen3-VL-4B
text encoder, so it reads your prompt as plain language. Describe the image the way you'd describe it to
a person; it is especially strong on **photographic realism, lighting, and coherent composition**.

> **Krea 2 Turbo** is the fast, few-step (~8) distilled variant shipped here — the model is trajectory-
> distilled (TDM) and runs classifier-free guidance internally, so it is **CFG-free** (no guidance
> slider, no negative prompt). Best for quick, high-quality iteration up to 2048².

## Prompt Shape

`subject + scene + composition + camera/framing + aesthetic/style`

Krea reads the whole prompt as intent, so natural descriptive language works better than a pile of
disconnected tags.

## How Krea Reads A Prompt (keep it faithful and concise)

The goal is a clear prompt, not a long one:

- **Already clear → barely change it.** A short, well-defined prompt ("a cup of coffee on a windowsill")
  needs little more than maybe one style or lighting word. Don't invent scenes, props, or mood the
  prompt didn't mention.
- **Be concise.** Short sentences; don't repeat ideas, pile up synonyms ("realistic, photographic,
  ultra-real"), or add empty praise ("stunning", "premium", "masterpiece"). Concrete quality words like
  `cinematic` or `soft window light` are fine.
- **Describe what you want, not what you don't.** Krea Turbo is CFG-free and takes **no negative
  prompt**, so avoid negations ("no people", "without text") — state the positive scene instead; a thing
  you don't name simply won't appear.

## Build The Prompt

### Subject

Name the main subject and its visible traits (color, material, clothing, expression). One or two clear
subjects render more coherently than a crowded frame.

### Scene Details

Describe background, foreground, lighting, weather, and time of day. Krea is particularly responsive to
**lighting description** — "soft overcast light", "golden-hour backlight", "hard noon sun" — which
drives much of its photorealism.

### Composition & Framing

Direct framing language works: `low angle`, `wide shot`, `medium close-up`, `centered subject`,
`rule of thirds`, `shallow depth of field`.

### Style

Concise style labels work well: `photorealistic`, `cinematic`, `editorial photography`, `watercolor`,
`flat illustration`, `studio portrait`, `film noir`. Name a known style by its name only (e.g.
`Kodak Portra`, `ukiyo-e`, `cyberpunk`) — you don't need to describe what it looks like. For everyday
realistic subjects you don't need to say "photorealistic"; that's already the default.

## Quality & Speed Notes

- **Resolution:** use a native bucket (1024² / 768×1024 / 1024×768 / 1280×720 / 720×1280 / 1536²); width
  and height must be multiples of 16. 1024² is the default. The model supports up to 2048².
- **Steps:** ~8 (few-step distilled — more steps rarely help and slow you down).
- **Guidance:** none — Krea Turbo is CFG-free (the guidance slider is inert).
- **Sampler / scheduler:** `default` is the native rectified-flow loop and the best starting point; the
  curated samplers/schedulers are exposed for experimentation.
- **Quantization:** Q8 is the default (near-lossless, ~27 GB peak — needs a 48 GB-class Mac); Q4 is a
  lighter option.

## Using LoRAs

Krea LoRAs train on the full **Krea 2 Raw** base and apply at inference on the distilled **Turbo**.
Because Turbo is few-step and CFG-free it adheres tightly to the prompt, so a LoRA's effect reads
softer here than it would on the Raw base. To compensate, Krea LoRAs start at a **higher default apply
weight (1.5)** than other families — they stay coherent well above that, so:

- If a LoRA's style or subject isn't coming through on a strongly-described scene, raise the weight
  toward **2.0** and leave the prompt a little room rather than over-specifying every detail.
- If a LoRA over-dominates, lower it. The weight slider is the main lever; the default is just a
  starting point tuned for the distilled Turbo.

## Example Prompts

`A weathered lighthouse on a rocky cliff at golden hour, waves breaking below, gulls in the distance.
Wide shot, low angle, warm directional light. Photorealistic, cinematic.`

`Portrait of an elderly fisherman, deep wrinkles, soft overcast window light, shallow depth of field.
Editorial photography.`

`A quiet Kyoto side street after rain at dusk, wet stone reflecting paper-lantern light, a lone cyclist.
Medium shot, centered. Cinematic, photorealistic.`

## Sources

- [Krea 2 (Hugging Face)](https://huggingface.co/krea)
- [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report)
- [Krea 2 Community License](https://www.krea.ai/krea-2-licensing)
