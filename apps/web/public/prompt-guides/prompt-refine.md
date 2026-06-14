# Prompt Refiner Guide

The Prompt Refiner is a small instruction LLM (Llama-3.2-3B) that powers the "Refine my prompt" control in Image and Video Studio. It rewrites your prompt to follow the selected generation model's own prompt guide — it does not generate images or video itself.

## What It Does

When you click **Refine my prompt**, your current prompt and the selected model's prompt guide are sent to this model. It returns a single rewritten prompt that preserves your intent (subjects, attributes, actions, setting) while tightening phrasing and adding only details that keep the result coherent and on-guide. You review the rewrite and choose **Apply** or **Keep original** — your prompt is never changed automatically.

## Installation

The refiner runs in-process on the native worker (MLX on macOS, candle on Windows/CUDA) and is **not** auto-downloaded. Install it once from the **Models** screen (it is also offered inline the first time you refine before it is present). It is about 7 GB and downloads into the shared Hugging Face cache, so other tools reuse it.

## Practical Notes

The refiner matches the language of your prompt: a non-English prompt is rewritten in the same language.

It works without a model guide, but the rewrite is most useful when the selected generation model ships one — the refiner follows that guide's recommended structure and what-to-avoid guidance.

If a prompt is already detailed and on-guide, the refiner makes only minimal edits for fluency.
