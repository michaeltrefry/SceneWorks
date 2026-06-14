# Bernini Image Prompt Guide

## Best For

High-quality **still images** where the scene has structure worth planning — multiple elements, a
described composition, or a specific mood. Bernini pairs a semantic planner (it interprets and lays
out your prompt before rendering) with a Wan2.2 renderer, so it rewards clear, descriptive prompts
more than terse ones.

> Bernini is heavy: the planner runs many passes before rendering, so a single image takes several
> minutes and a lot of unified memory. For everyday stills the fast native image models (Z-Image,
> FLUX, SDXL, Kolors, Lens, Chroma, Qwen-Image, SenseNova) are quicker. Reach for Bernini when you
> want planner-driven composition or to stay consistent with a Bernini video look.

## Task Modes

| Mode | What it does | Inputs you provide |
|---|---|---|
| **Text → Image** | Generates a single image from the prompt alone. | Prompt |
| **Image → Image (Edit)** | Re-renders a source image to follow your prompt (restyle, relight, change the setting) while keeping its overall structure. | Source image + prompt |

Tips per mode:

- **Image → Image:** the source sets the composition and structure, so write the prompt about *what
  changes* (the new style, lighting, wardrobe, palette, or setting), not the layout the image already
  has. A higher edit strength moves further from the source; a lower one stays closer.
- Both modes share the same resolution limits. Use a native bucket and let the renderer align to the
  patch stride (16).

## Prompt Shape

`subject + scene + composition + camera/framing + aesthetic/style`

Because the planner reads the whole prompt as intent, you can describe the image the way you'd brief a
photographer — what is in frame, how it's composed, how it's lit — and let Bernini organize it.

## Build The Prompt

### Subject

Name the main entity and its visible traits (color, material, clothing, expression). One or two clear
subjects render more coherently than a crowded frame.

### Scene Details

Describe background, foreground, lighting, weather, and time of day. A focused scene reads cleaner.

### Composition & Framing

Bernini responds to direct framing language (inherited from the Wan2.2 renderer):

- `low angle`, `wide shot`, `medium close-up`, `centered subject`, `rule of thirds`
- `shallow depth of field`, `bokeh background`

### Style

Concise style labels work well: `cinematic`, `photorealistic`, `warm commercial`, `documentary`,
`film noir`, `studio portrait`.

### Negative Prompt

Reduce common artifacts: `blurry details, subtitles, low quality, distorted hands, extra limbs,
watermark, jpeg artifacts, crowded background`.

## Quality & Speed Notes

- **Resolution:** use a native bucket (512² / 768² / 1024² / 1280×720 / 720×1280). Very small frames
  look degraded.
- **Quantization:** Q4 is the default (fits ~64 GB Macs, faster). Opt into Q8 (advanced setting) for a
  small quality gain if you have the memory.
- **Steps:** more steps = cleaner detail but longer renders; keep them modest for iteration.
- **Count:** each image is minutes — keep batches small.

## Example Prompts

`A weathered lighthouse on a rocky cliff at golden hour, waves breaking below, gulls in the distance.
Wide shot, low angle, warm directional light, shallow depth of field. Photorealistic, cinematic.`

`Portrait of an elderly luthier in his workshop, soft window light, wood shavings on the bench,
focused expression. Medium close-up, rule of thirds. Documentary, muted palette.`

## Sources

- [Bernini model card](https://huggingface.co/ByteDance/Bernini-Diffusers)
- [Bernini GitHub repo](https://github.com/bytedance/Bernini)
- [Wan2.2 GitHub repo](https://github.com/Wan-Video/Wan2.2)
