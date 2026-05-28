# InstantID (RealVisXL) Prompt Guide

## Best For

Generating **the same person** across many different scenes, poses, and outfits from a **single reference image** — no LoRA training required. InstantID reads a face from your character's approved reference (an insightface ArcFace embedding + 5-point facial landmarks) and locks identity onto a photoreal RealVisXL render while your prompt drives everything else.

Use it when you want faithful likeness *and* scene freedom — the combination plain IP-Adapter ("reference strength" copy) can't deliver. It is the bridge for **created characters**: build an identity-consistent set of images first, then (optionally) train a per-character LoRA from them.

This is a **reference-driven** model: it only runs in the "With character" flow and always needs a clear reference face. There is no plain text-to-image or edit mode.

## How It Works

- **Identity comes from the reference image, not the prompt.** Do *not* describe the person's face, hair color, or features — the model takes those from the reference. Describing appearance only fights the reference.
- **The prompt drives the scene:** setting, action, pose, framing, wardrobe, lighting, and style.
- **Reference strength** (the slider) controls how hard identity is pinned. Higher = closer likeness but stiffer; lower = more natural and prompt-flexible.

## View Angle

The head angle does **not** come from the prompt — describing "profile" or "looking left" won't rotate the face, because identity pins it toward the reference's angle. Instead use the **View angle** dropdown: front, three-quarter left/right, left/right profile, looking up, looking down, and the four diagonals. Each renders the *same* character at that angle with identity preserved (validated ~0.81–0.89 likeness across all of them). View-angle renders square. Leave it on **Match reference** to keep the reference's own angle. Generating one character across several angles is also how you build a consistent set for training a character LoRA.

## Pose Library

The **Pose library** picker (in Character Studio and Image Studio) generates the character in poses you choose from a bundled gallery — standing, sitting, dancing, jumping, and more. Tick one or more poses and generate; you get one image per pose in a single job, sharing one seed so hair/clothing stay consistent. An **OpenPose ControlNet** drives the pose from the pose's skeleton; the prompt should describe the **outfit and shoes** (InstantID holds the face, not the wardrobe) — leave stance words out, the skeleton sets the pose. Because the face is small at full-body framing, each image can add a **face-restoration pass** (the face is detected, upscaled, re-rendered with the character's identity, and blended back in) that recovers likeness to ~0.88 cosine. It's controlled by the **Restore face** toggle (on by default): turn it off to use the raw OpenPose+InstantID render with no blend artifacts — a cleaner composite, but weaker identity at small face sizes. Poses where the face isn't visible (e.g. from behind) rely on the shared seed for continuity. With restoration on, each pose is two renders, so larger selections take longer and use more memory.

## Choose A Good Reference

Identity quality is set by the reference more than the prompt:

- A **clear, front-facing** photo where the face is large and well-lit.
- **One** unobstructed face (no sunglasses, heavy shadow, or extreme profile).
- Sharp focus, neutral expression works best as a baseline.

A side profile, tiny face, or low-light crop will weaken the likeness no matter how good the prompt is.

## Build The Prompt

Front-load the scene and action, then layer style and lighting. Leave the face to the reference.

### Scene & Action

`sitting at a sidewalk cafe in the morning, holding a coffee cup`

`walking through a rain-slick neon city street at night`

### Wardrobe

`wearing a tailored charcoal wool coat` · `in a worn denim jacket` · `in athletic running gear`

### Framing & Camera

`candid 35mm photograph, shallow depth of field` · `medium portrait, eye-level` · `wide environmental shot`

### Lighting & Style

`soft natural window light` · `golden hour backlight` · `cinematic film still` · `editorial fashion photography`

## Negative Prompts

RealVisXL honors a negative prompt — use it to push away the plastic look InstantID can drift toward:

`plastic skin, airbrushed, cgi, 3d render, cartoon, illustration, anime, waxy, overprocessed, deformed, extra fingers, watermark, text`

## Tips

- **Don't describe the face.** Hair color, eye color, and features come from the reference; restating them only adds noise.
- **Reference strength ~0.6–0.8.** Start mid-range. Raise it if the likeness drifts; lower it (≤0.5) for more natural skin and looser, more prompt-driven results.
- **~30 steps at guidance 5.0** is the validated baseline. Lower guidance (3.5–5) reads more photographic; higher pushes prompt adherence at the cost of "baked" skin.
- **Explore takes:** raise Variations and leave the seed blank to get different poses/scenes of the same person.
- **Identity is idealized.** InstantID tends to glamorize (smoother skin, lighter makeup, fewer freckles). That's expected — great for invented characters, a mild flattering bias for real ones.
- **First run downloads** the InstantID weights (~4GB) and the antelopev2 face pack on top of the RealVisXL checkpoint; later runs reuse them.

## Example Prompts

`Candid photograph at a sidewalk cafe in the morning, holding a coffee cup, wearing a denim jacket, soft natural light, 35mm, shallow depth of field, photorealistic.`

`Cinematic film still, standing on a rain-slick neon city street at night, leather jacket, reflections on wet pavement, moody rim lighting, shallow depth of field.`

## Sources

- [InstantID project](https://github.com/instantX-research/InstantID)
- [InstantID weights](https://huggingface.co/InstantX/InstantID)
- [RealVisXL_V5.0 model card](https://huggingface.co/SG161222/RealVisXL_V5.0)
