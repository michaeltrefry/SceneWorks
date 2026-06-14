# Bernini Prompt Guide

## Best For

High-quality short **text-to-video** clips where the scene has structure worth planning — multiple
elements, a described action, or a specific mood. Bernini pairs a semantic planner (it interprets and
lays out your prompt before rendering) with a Wan2.2 video renderer, so it rewards clear, descriptive
prompts more than terse ones.

> Bernini is heavy: the planner runs many passes before rendering, so a clip takes several minutes.
> Keep clips short and write motion that completes within the selected duration.

## Prompt Shape

`subject + scene + motion + camera + aesthetic/style`

Because the planner reads the whole prompt as intent, you can describe the scene the way you'd brief a
cinematographer — what is in frame, what happens, how it's shot — and let Bernini organize it.

## Build The Prompt

### Subject

Name the main entity and its visible traits (color, material, clothing, expression). One or two clear
subjects render more coherently than a crowded frame.

### Scene Details

Describe background, foreground, lighting, weather, and time of day. A focused scene keeps motion
stable across the clip.

### Motion

State what moves, how much, and how fast — the action should fit the clip length:

- `slowly turns toward the window`
- `walking forward across the meadow`
- `leaves drifting on a gentle breeze`

### Camera

Bernini responds to direct camera language (inherited from the Wan2.2 renderer):

- `fixed camera`, `camera pushes in`, `camera pans left`
- `low angle`, `wide shot`, `medium close-up`

### Style

Concise style labels work well: `cinematic`, `photorealistic`, `warm commercial video`,
`documentary`, `film noir`.

### Negative Prompt

Reduce common video artifacts: `static frame, blurry details, subtitles, low quality, distorted hands,
extra limbs, flicker, crowded background`.

## Quality & Speed Notes

- **Resolution:** use a native bucket (480×848 / 848×480 / 720p). Very small frames look degraded.
- **Quantization:** Q4 is the default (fits ~64 GB Macs, faster). Opt into Q8 (advanced setting) for a
  small quality gain if you have the memory.
- **Steps:** more steps = cleaner motion but longer renders; keep them modest for iteration.

## Example Prompts

`A golden retriever puppy runs across a sunlit meadow toward the camera, ears bouncing, soft morning
light and a shallow depth of field. The camera holds steady at a low angle. Photorealistic, warm,
gentle motion.`

`A lone astronaut walks across a red desert at dusk, dust trailing behind each step. Wide shot, the
camera slowly pushes in as the wind picks up. Cinematic, muted palette, quiet and vast.`

## Sources

- [Bernini model card](https://huggingface.co/ByteDance/Bernini-Diffusers)
- [Bernini GitHub repo](https://github.com/bytedance/Bernini)
- [Wan2.2 GitHub repo](https://github.com/Wan-Video/Wan2.2)
