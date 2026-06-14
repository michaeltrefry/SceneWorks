# JoyCaption Captioning Guide

JoyCaption is a vision-language model used by **training-dataset captioning** ("Caption images" / "Re-caption"). It looks at each image in a dataset and writes a caption — it does not generate images or video. Good captions help a LoRA learn the right concepts, so this step matters for training quality.

## Installation

JoyCaption runs in-process on the native worker (MLX on macOS, candle on Windows/CUDA) and is **not** auto-downloaded. Install it once from the **Models** screen (the caption dialog also offers a download when it's missing). It is about 17 GB and downloads into the shared Hugging Face cache, so other tools reuse it.

## Caption Settings

**Type** picks the caption style (e.g. Descriptive). **Length** trades brevity against detail — shorter captions for tight concept LoRAs, longer for rich scene description.

**Character name** injects a consistent name/trigger into captions, useful for character LoRAs.

**Caption prompt** is the instruction sent to the model; leave it blank to use the type/length defaults, or override it for full control.

## Practical Notes

Captioning runs as a background job — queue it from the dataset editor and track progress in the Queue.

"Re-caption" overwrites existing captions; leave it off to only fill in images that have no caption yet.

Review and edit captions before training — automatic captions are a fast first pass, not a substitute for a quick human check.
