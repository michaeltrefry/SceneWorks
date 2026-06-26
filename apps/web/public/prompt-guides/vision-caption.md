# Vision Captioner Guide

The Vision Captioner is a vision-language model that turns a **reference image** into a structured Ideogram-style JSON caption — a style description plus a grounded list of the elements it sees (with bounding boxes). That caption feeds image-to-prompt **variations**: you give it a picture, it describes the style and composition, and that description becomes a prompt you can regenerate from. It does not generate images itself.

The default model is an abliterated **Qwen3-VL-8B-Instruct**.

## Installation

The Vision Captioner runs in-process on the native MLX worker (Apple Silicon only for now). It is **not** auto-downloaded — install it once from the **Models** screen. It is about 18 GB and downloads into the shared Hugging Face cache, so other tools reuse the snapshot. A 64 GB-class Mac is recommended.

## How It Works

Point it at a reference image and it observes what is actually in the frame — it describes the style and the visible elements rather than inventing content. The output is JSON: a `style_description` plus grounded elements with bounding boxes, in the same shape Ideogram's prompt surface uses. The model is steered to emit valid JSON, and malformed output is rejected and retried.

## Practical Notes

The goal is **style and composition variations**, not a pixel-perfect reconstruction of the source. Expect the caption to capture the look, mood, palette, and layout — then vary the specifics when you regenerate.

Captioning is a fast first pass. Review the generated caption and tweak it before generating if you want to push the result in a particular direction.
