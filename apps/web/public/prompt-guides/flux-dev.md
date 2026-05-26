# FLUX.1 [dev] Prompt Guide

## Best For

Highest-quality FLUX text-to-image: detailed photography, illustration, concept art, and crisp, legible text. [dev] is the guided ~28-step variant — slower than [schnell] but with finer detail and guidance control.

> **License:** FLUX.1 [dev] is distributed under the FLUX.1 [dev] Non-Commercial License — generations are for non-commercial use only. It is a gated Hugging Face download (accept the license and add a Hugging Face token under Settings → Service credentials first). For commercial use, choose FLUX.1 [schnell] (Apache-2.0).

## Prompt Shape

FLUX was trained on rich natural-language captions, so write a fluent sentence (or two) describing the finished image rather than a list of tags:

`subject + setting + visual details + style + composition + lighting + any text`

[dev] uses an embedded **guidance** value (default ~3.5): higher values follow the prompt more literally, lower values are more creative. There is no separate negative prompt — keep everything in the positive prompt.

## Build The Prompt

### Subject

Describe the subject as if captioning a finished photo. Include type, action, position, and distinguishing features.

Good: `a botanist examining a glowing bioluminescent orchid in a misty greenhouse`

### Details

Add material, texture, and atmosphere:

- `brushed titanium`
- `dew on spider silk`
- `volumetric haze`
- `ornate art-nouveau scrollwork`

### Style

Use photographic or art-direction terms:

- `National Geographic wildlife photography`
- `cinematic film still, anamorphic lens`
- `detailed digital matte painting`
- `loose watercolor on cold-press paper`

### Camera And Composition

FLUX handles many aspect ratios — pair composition with the selected size:

- `overhead flat-lay`
- `telephoto compression`
- `Dutch angle`
- `symmetrical centered composition`

### Lighting

- `golden hour`
- `soft chiaroscuro`
- `bioluminescent glow`
- `hard noon sunlight with sharp shadows`

### Text In Images

FLUX renders text well — quote the exact words and describe the medium:

`a letterpress poster reading "NORTHERN LIGHTS FESTIVAL" in condensed sans-serif, layered ink texture`

## Tips

- Start at guidance ~3.5; raise toward 5 for stricter prompt adherence, lower toward 2 for more variation.
- ~28 steps balances quality and speed; more steps add marginal detail.
- No negative prompt — describe what you *want*, in detail.

## Example Prompts

`A grand library reading room at golden hour, sun shafts through tall arched windows, dust motes in the air, a brass plaque reading "QUIET STUDY" on a carved oak desk, leather-bound books, intricate coffered ceiling, cinematic depth.`

`An ultra-detailed macro photograph of a dew-covered jumping spider on a fern frond at dawn, iridescent eyes, soft bokeh background, National Geographic wildlife photography, razor-sharp focus.`

## Sources

- [FLUX.1 [dev] model card](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Diffusers Flux pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)
