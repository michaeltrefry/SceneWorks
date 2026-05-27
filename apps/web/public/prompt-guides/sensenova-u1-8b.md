# SenseNova-U1 8B Prompt Guide

## Best For

Dense images, infographics, posters, documents, precise layouts, image editing, and multimodal tasks where text and visual structure both matter.

## Prompt Shape

Use a structured design brief:

`purpose + subject/content + layout + visual hierarchy + exact text + style + camera/composition + details`

Short prompts can under-constrain SenseNova-U1, especially for infographics. Expand simple ideas into a clear layout and content plan.

## Build The Prompt

### Subject

State the topic and main visual subject. For informational images, define the message the viewer should understand.

Good: `an infographic explaining how urban rain gardens reduce street flooding`

### Details

For dense layouts, specify sections, labels, icons, charts, captions, and reading order. Keep every required text string exact and in quotes.

### Style

Name the design language:

- `clean educational infographic`
- `flat vector poster`
- `arXiv-style technical page`
- `presentation slide`
- `comic explainer`

### Camera And Composition

For generated images, use normal camera terms. For design layouts, use layout terms:

- `two-column layout`
- `four-card grid`
- `large title at the top`
- `central diagram with callout labels`
- `clear margins and no overlapping text`

### Lighting

For realistic scenes, describe lighting. For graphics, describe color palette and contrast instead.

### Text And Typography

SenseNova-U1 is useful for dense text, but it still needs exact instructions. Include font feel, relative size, alignment, and hierarchy.

### Editing

For edits, say what to preserve first, then what to change. Keep the edit instruction concrete and ordered.

### Avoid

- Vague one-line prompts for infographics or posters — they under-constrain the layout.
- Approximate or unquoted text; always quote the exact strings you want rendered.
- Conflicting or overlapping layout instructions (e.g. "centered" and "left-aligned" for the same element).
- Cramming too many competing sections into one image; fewer, clearly bounded sections render more reliably.
- Generic quality tags like `masterpiece` or `best quality`; describe concrete content and structure instead.

## Example Prompts

`Create a vertical educational infographic titled "RAIN GARDENS AT WORK". Use a clean flat vector style with a blue and green palette. Top section: a city street with rain falling. Middle section: a cutaway soil diagram with arrows labeled "runoff", "plant roots", and "filtered water". Bottom section: three benefit cards reading "Less flooding", "Cleaner rivers", and "More habitat". Large readable sans-serif text, clear margins, no overlapping elements.`

`A realistic product photograph of a transparent smart thermostat on a white wall, showing the screen text "72 F" in crisp blue digits. Soft morning window light from the left, minimal interior background, centered composition, shallow depth of field, polished glass reflections.`

## Sources

- [SenseNova-U1 model card](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT)
- [SenseNova-U1 prompt enhancement doc](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT/blob/main/docs/prompt_enhancement.md)
- [SenseNova-U1 Infographic model card](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic)
