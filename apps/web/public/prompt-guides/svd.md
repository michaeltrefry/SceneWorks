# Stable Video Diffusion Guide

## Best For

Animating a single still image into a short, natural-motion clip — subtle camera moves, drifting elements, living portraits, looping ambiance. Stable Video Diffusion (img2vid-XT) is **image-conditioned only**: it takes one source image and produces a ~25-frame clip. There is **no text prompt** — the motion comes from the image and the motion controls, not from words.

## Licensing

SVD ships under the **Stability AI Community License**: commercial use is free under $1M annual revenue (paid Enterprise above), and the weights are ungated (no Hugging Face token needed). SceneWorks downloads the weights for local use; it does not redistribute them. Review Stability's Acceptable Use Policy before commercial use.

## How To Use It

1. **Pick a strong source image.** SVD animates what it's given — composition, subject, and lighting all carry into the clip. Sharp, well-exposed images at the native aspect ratio work best.
2. **Match the resolution.** SVD-XT is trained at **1024×576** (landscape) and **576×1024** (portrait). Pick the orientation that matches your image; off-ratio inputs get letterboxed or cropped.
3. **Generate.** The model produces a fixed ~25-frame burst. Set the playback fps to pace it (lower fps = slower, more deliberate motion).

## Motion Controls (Advanced)

SVD's "prompt" is really a small set of conditioning knobs, available under advanced settings:

- **Motion strength** (`motion bucket`) — how much movement the model introduces. Lower = subtle drift; higher = bold motion (and more risk of artifacts). The default is a balanced mid-range value.
- **Conditioning fps** — the frame rate the model conditions on (separate from playback fps). Lower values read as slower, smoother motion.
- **Noise augmentation** — how far the first frame is allowed to drift from the source. Small values keep the opening frame faithful to your image.

## Tips

- Start from a high-quality image — SVD amplifies whatever is already there, including blur and noise.
- Keep motion strength moderate for portraits and product shots; raise it for landscapes and abstract motion.
- SVD has no prompt and no negative prompt — to change the result, change the source image or the motion controls.
- The clip length is fixed by the model (~25 frames); use playback fps to control pacing rather than expecting longer durations.

## Sources

- [SVD img2vid-XT model card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
- [Stability AI Community License](https://stability.ai/community-license-agreement)
- [Diffusers SVD pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/svd)
