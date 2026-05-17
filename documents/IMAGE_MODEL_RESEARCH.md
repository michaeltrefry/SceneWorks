# Image Model Research

## Recommendation

Use `z_image_turbo` as the first SceneWorks image-generation target, keep `z_image_edit` as the matching edit target, and keep `qwen_image_edit` available behind the same adapter boundary for higher-control edit workflows.

## Rationale

- Z-Image-Turbo is a 6B image model with an official Hugging Face/Diffusers path, Apache-2.0 licensing, an 8-step distilled target, and stated fit for 16GB VRAM consumer devices. That makes it the best first adapter target for a local app that should feel responsive on 24GB+ GPUs.
- Z-Image-Turbo is text-to-image oriented, so SceneWorks should not pretend it can edit source images. The app now routes edit workflows only to edit-capable manifest entries.
- Qwen Image remains the strongest follow-on edit family. The official Qwen repository describes Qwen-Image as a 20B model family with strong text rendering and precise image editing, and the Hugging Face model card exposes a `QwenImageEditPipeline`.
- Qwen-Image-Edit-2509 is worth evaluating next for better edit consistency and multi-image input, but it should not block the first Image Studio vertical slice.

## Sources

- Z-Image-Turbo model card: https://huggingface.co/foyoux/Z-Image-Turbo
- Qwen Image repository: https://github.com/QwenLM/Qwen-Image
- Qwen-Image-Edit model card: https://huggingface.co/Qwen/Qwen-Image-Edit
- Diffusers Qwen Image documentation: https://huggingface.co/docs/diffusers/api/pipelines/qwenimage

## Implementation Note

This epic lands the adapter seam, model manifests, job payloads, generated asset sidecars, GenerationSet records, recipes, review UI, and project library integration. The current worker uses a deterministic procedural renderer so the full product flow works without downloading model weights. A future `z_image` runtime can replace the renderer inside the worker adapter without changing frontend or API contracts.
