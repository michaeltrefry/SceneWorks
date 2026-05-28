// Job-status/job-type/quality enums live in jobTypes.js (single source of truth,
// sc-1657). Re-exported here so existing `from "./constants.js"` importers are
// unaffected.
export { terminalStatuses, actionStatuses } from "./jobTypes.js";

// SenseNova-U1 interleave resolution buckets (distinct from plain text-to-image).
// Mirrors the worker's interleave_resolution_for / upstream examples/interleave.
export const INTERLEAVE_RESOLUTION_OPTIONS = [
  "1536x1536",
  "2048x1152",
  "1152x2048",
  "1888x1248",
  "1248x1888",
  "1760x1312",
  "1312x1760",
];
export const DEFAULT_INTERLEAVE_RESOLUTION = "2048x1152";

// Default interleave system prompt (the think/no-think protocol). Prefilled in
// Document Studio; the worker falls back to this same text when the field is blank.
// Keep in sync with apps/worker/scene_worker/image_adapters.py::_INTERLEAVE_SYSTEM_MESSAGE.
export const DEFAULT_INTERLEAVE_SYSTEM_MESSAGE = `You are a multimodal assistant capable of reasoning with both text and images. You support two modes:

Think Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.

Non-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.

After the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer.`;
export const fallbackModels = [
  {
    id: "z_image_turbo",
    name: "Z-Image-Turbo",
    type: "image",
    // No `character_image`: the worker adapter has no IP-Adapter wiring (sc-2005).
    capabilities: ["text_to_image", "style_variations"],
    ui: {
      description: "Fast local text-to-image target.",
      promptGuide: { title: "Z-Image-Turbo Prompt Guide", path: "/prompt-guides/z-image-turbo.md" },
    },
  },
  {
    id: "qwen_image",
    name: "Qwen Image",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: {
      description: "Qwen text-to-image target.",
      promptGuide: { title: "Qwen Image Prompt Guide", path: "/prompt-guides/qwen-image.md" },
    },
  },
  {
    id: "z_image_edit",
    name: "Z-Image-Edit",
    type: "image",
    capabilities: ["edit_image"],
    ui: {
      description: "Image edit target.",
      promptGuide: { title: "Z-Image-Edit Prompt Guide", path: "/prompt-guides/z-image-edit.md" },
    },
  },
  {
    id: "qwen_image_edit",
    name: "Qwen Image Edit",
    type: "image",
    // character_image (sc-2014): subject consistency via QwenImageEditPipeline's
    // dual-control architecture; reference goes through `image=` + trueCfgScale.
    capabilities: ["edit_image", "character_image"],
    ui: {
      description: "Qwen image edit target. Dual-control architecture (semantic + appearance) handles both localized edits and subject-consistency across new scenes/poses (Character Studio reference). Apache-2.0, ungated.",
      promptGuide: { title: "Qwen Image Edit Prompt Guide", path: "/prompt-guides/qwen-image-edit.md" },
    },
  },
  {
    id: "qwen_image_edit_2509",
    name: "Qwen Image Edit (2509)",
    type: "image",
    capabilities: ["edit_image", "character_image"],
    ui: {
      description: "September monthly iteration of Qwen-Image-Edit (Qwen/Qwen-Image-Edit-2509) via QwenImageEditPlusPipeline. Enhanced subject-consistency for character-in-new-context generation; multi-image reference support. Apache-2.0, ungated; ~50 steps at trueCfgScale 4.0.",
      promptGuide: { title: "Qwen Image Edit (2509) Prompt Guide", path: "/prompt-guides/qwen-image-edit-2509.md" },
    },
  },
  {
    id: "lens",
    name: "Lens",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: {
      description: "Microsoft Lens base text-to-image (20-step, CFG 5.0); higher quality than Turbo, also the LoRA training base.",
      promptGuide: { title: "Lens Prompt Guide", path: "/prompt-guides/lens.md" },
    },
  },
  {
    id: "lens_turbo",
    name: "Lens-Turbo",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: {
      description: "Microsoft Lens distilled 4-step text-to-image; strong text rendering, large-VRAM GPU.",
      promptGuide: { title: "Lens-Turbo Prompt Guide", path: "/prompt-guides/lens-turbo.md" },
    },
  },
  {
    id: "sensenova_u1_8b",
    name: "SenseNova-U1 8B",
    type: "image",
    capabilities: ["text_to_image", "edit_image", "vqa", "interleave"],
    limits: { resolutions: ["2048x2048", "2720x1536", "2496x1664", "2368x1760", "1536x2720", "1664x2496", "1760x2368"] },
    ui: {
      description: "Unified multimodal model (NEO-unify, ~16B); native text-to-image and instruction editing with strong text rendering and infographics. Heavy (~42GB bf16); CUDA or 96GB+ Apple Silicon.",
      promptGuide: { title: "SenseNova-U1 8B Prompt Guide", path: "/prompt-guides/sensenova-u1-8b.md" },
    },
  },
  {
    id: "sensenova_u1_8b_fast",
    name: "SenseNova-U1 8B Fast",
    type: "image",
    capabilities: ["text_to_image", "edit_image"],
    limits: { resolutions: ["2048x2048", "2720x1536", "2496x1664", "2368x1760", "1536x2720", "1664x2496", "1760x2368"] },
    ui: {
      description: "8-step distilled SenseNova-U1; ~5-6x faster text-to-image and editing (~50s/image on MPS) at a small quality trade-off. Shares the base 8B weights; a ~0.4GB distill LoRA downloads automatically. Distilled editing is experimental — use the base model for max-quality edits.",
      promptGuide: { title: "SenseNova-U1 8B Fast Prompt Guide", path: "/prompt-guides/sensenova-u1-8b-fast.md" },
    },
  },
  {
    id: "flux_schnell",
    name: "FLUX.1 [schnell]",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: {
      description: "FLUX.1 [schnell] — fast ~4-step distilled text-to-image, Apache-2.0 (commercial-safe). ~34GB bf16, large-VRAM GPU.",
      promptGuide: { title: "FLUX.1 [schnell] Prompt Guide", path: "/prompt-guides/flux-schnell.md" },
    },
  },
  {
    id: "flux_dev",
    name: "FLUX.1 [dev]",
    type: "image",
    // character_image: XLabs FLUX IP-Adapter (sc-2011 resemblance tier).
    capabilities: ["text_to_image", "character_image", "style_variations"],
    ui: {
      description: "FLUX.1 [dev] — higher-quality ~28-step text-to-image under the FLUX.1 [dev] Non-Commercial License (non-commercial only); gated download needs an HF token + license acceptance. ~34GB bf16, large-VRAM GPU. With a character reference, runs XLabs IP-Adapter for scene-flexible resemblance (faithful identity belongs to PuLID-FLUX).",
      promptGuide: { title: "FLUX.1 [dev] Prompt Guide", path: "/prompt-guides/flux-dev.md" },
    },
  },
  {
    id: "chroma1_hd",
    name: "Chroma1-HD",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: {
      description: "Chroma1-HD — high-resolution text-to-image, Apache-2.0 (commercial-safe). FLUX.1-schnell-derived 8.9B + T5-XXL; true CFG with negative prompts (~40 steps, guidance 3.0). Large-VRAM GPU.",
      promptGuide: { title: "Chroma1-HD Prompt Guide", path: "/prompt-guides/chroma1-hd.md" },
    },
  },
  {
    id: "chroma1_base",
    name: "Chroma1-Base",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: {
      description: "Chroma1-Base — text-to-image foundation tuned for finetuning, Apache-2.0 (commercial-safe). FLUX.1-schnell-derived 8.9B + T5-XXL; true CFG with negative prompts (~40 steps, guidance 3.0). Large-VRAM GPU.",
      promptGuide: { title: "Chroma1-Base Prompt Guide", path: "/prompt-guides/chroma1-base.md" },
    },
  },
  {
    id: "chroma1_flash",
    name: "Chroma1-Flash",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: {
      description: "Chroma1-Flash — fast CFG-baked text-to-image, Apache-2.0 (commercial-safe). FLUX.1-schnell-derived 8.9B + T5-XXL; ~8-step generation, CFG disabled (guidance 1.0, no negative prompt). Large-VRAM GPU.",
      promptGuide: { title: "Chroma1-Flash Prompt Guide", path: "/prompt-guides/chroma1-flash.md" },
    },
  },
  {
    id: "kolors",
    name: "Kolors",
    type: "image",
    capabilities: ["text_to_image", "edit_image", "character_image", "style_variations"],
    ui: {
      description: "Kwai-Kolors Kolors — photorealistic text-to-image with strong Chinese + English prompting and text rendering. Apache-2.0 (commercial-safe). ChatGLM3-6B + SDXL-style UNet; ~16.5GB, real CFG + negative prompt, ~25 steps at guidance 5.0.",
      promptGuide: { title: "Kolors Prompt Guide", path: "/prompt-guides/kolors.md" },
    },
  },
  {
    id: "sdxl",
    name: "Stable Diffusion XL",
    type: "image",
    capabilities: ["text_to_image", "edit_image", "character_image", "style_variations"],
    ui: {
      description: "Stability AI Stable Diffusion XL base 1.0 — open text-to-image foundation with the largest LoRA/finetune ecosystem. CreativeML OpenRAIL++-M (commercial use OK, ungated). SDXL UNet + dual CLIP; ~6.9GB fp16, real CFG + negative prompt, ~30 steps at guidance 7.0; native 1024x1024. With a character reference, runs IP-Adapter plus-face for scene-flexible resemblance (faithful likeness — see InstantID).",
      promptGuide: { title: "Stable Diffusion XL Prompt Guide", path: "/prompt-guides/sdxl.md" },
    },
  },
  {
    id: "realvisxl",
    name: "RealVisXL (photoreal SDXL)",
    type: "image",
    capabilities: ["text_to_image", "edit_image", "character_image", "style_variations"],
    ui: {
      description: "Photoreal SDXL finetune that targets the \"shiny/plastic\" look of base SDXL — the same RealVisXL_V5.0 checkpoint the InstantID built-in uses, exposed as a plain selectable. openrail++ (commercial use OK, ungated). Same SDXL UNet + dual CLIP, sdxl-family LoRA support, real CFG + negative prompt; ~30 steps at guidance 7.0, native 1024x1024. With a character reference, runs IP-Adapter plus-face for scene-flexible resemblance.",
      promptGuide: { title: "RealVisXL Prompt Guide", path: "/prompt-guides/realvisxl.md" },
    },
  },
  {
    id: "instantid_realvisxl",
    name: "InstantID (RealVisXL)",
    type: "image",
    // Reference-driven only — appears solely in the "With character" picker.
    capabilities: ["character_image"],
    ui: {
      description: "Identity-preserving character generation — holds a person's face from a single reference image while the prompt drives scene, pose, and wardrobe. RealVisXL_V5.0 (photoreal SDXL, openrail++ commercial-OK) + InstantID ArcFace embedding & landmark ControlNet; faithful likeness with scene freedom (vs IP-Adapter resemblance only). Pick a character with an approved reference, then raise Variations. ~30 steps at guidance 5.0, ~22GB peak.",
      promptGuide: { title: "InstantID (RealVisXL) Prompt Guide", path: "/prompt-guides/instantid-realvisxl.md" },
      // Identity tuning: reference strength (ipAdapterScale) defaults higher for
      // InstantID; identityStructure adds the controlnetConditioningScale slider.
      referenceStrengthDefault: 0.8,
      identityStructure: { label: "Identity structure", default: 0.8, min: 0.3, max: 1.0, step: 0.05 },
      // Canonical head angles (advanced.viewAngle; built-in landmark pack drives pose).
      viewAngles: [
        { id: "three_quarter_left", label: "Three-quarter left" },
        { id: "three_quarter_right", label: "Three-quarter right" },
        { id: "left_profile", label: "Left profile" },
        { id: "right_profile", label: "Right profile" },
        { id: "up", label: "Looking up" },
        { id: "down", label: "Looking down" },
        { id: "up_left", label: "Up · left" },
        { id: "up_right", label: "Up · right" },
        { id: "down_left", label: "Down · left" },
        { id: "down_right", label: "Down · right" },
        { id: "front", label: "Front" },
      ],
      // Pose library (advanced.poses): generate the character in poses chosen from the
      // bundled OpenPose gallery. Gates the pose picker in Character/Image Studio.
      poseLibrary: true,
    },
  },
  {
    id: "ltx_2_3",
    name: "LTX-2.3",
    type: "video",
    capabilities: ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
    defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
    limits: {
      durations: [4, 6, 8, 10, 12, 15],
      recommendedMaxDuration: 10,
      fps: [24, 25, 30],
      resolutions: ["768x512", "512x768", "640x640", "1280x720", "720x1280"],
    },
    ui: {
      description: "First-class short-shot video target.",
      durationHint: "Best at 10s or less for the current workflow.",
      promptGuide: { title: "LTX-2.3 Prompt Guide", path: "/prompt-guides/ltx-2-3.md" },
    },
  },
  {
    id: "ltx_2_3_eros",
    name: "LTX-2.3 10Eros",
    type: "video",
    capabilities: ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
    defaults: { duration: 6, fps: 25, resolution: "768x512", quality: "balanced" },
    limits: {
      durations: [4, 6, 8, 10, 12, 15],
      recommendedMaxDuration: 10,
      fps: [24, 25, 30],
      resolutions: ["768x512", "512x768", "640x640", "1280x720", "720x1280"],
    },
    ui: {
      description: "Community LTX-2.3 merge tuned for image-to-video; uses LTX-video LoRAs.",
      durationHint: "Best at 10s or less for the current workflow.",
      promptGuide: { title: "LTX-2.3 10Eros Prompt Guide", path: "/prompt-guides/ltx-2-3-10eros.md" },
    },
  },
  {
    id: "svd",
    name: "Stable Video Diffusion",
    type: "video",
    capabilities: ["image_to_video"],
    // Image-conditioned only — no text prompt (Video Studio drops the prompt requirement).
    promptless: true,
    defaults: { duration: 4, fps: 7, resolution: "1024x576", quality: "balanced" },
    limits: {
      durations: [4],
      recommendedMaxDuration: 4,
      fps: [6, 7, 8, 10, 12, 25],
      resolutions: ["1024x576", "576x1024"],
    },
    ui: {
      description: "Stable Video Diffusion (img2vid-XT) — animates a source image into a short ~25-frame clip; no text prompt. Stability AI Community License (commercial free under $1M revenue, ungated).",
      durationHint: "Fixed ~25-frame clip from one image; adjust playback fps for pacing.",
      promptGuide: { title: "Stable Video Diffusion Guide", path: "/prompt-guides/svd.md" },
    },
  },
  {
    id: "wan_2_2",
    name: "Wan2.2",
    type: "video",
    capabilities: ["image_to_video", "text_to_video", "first_last_frame", "extend_clip", "video_bridge", "replace_person"],
    defaults: { duration: 5, fps: 24, resolution: "1280x720", quality: "balanced" },
    limits: {
      durations: [4, 5, 6, 7, 8],
      recommendedMaxDuration: 7,
      fps: [16, 24],
      resolutions: ["832x480", "1280x720", "720x1280"],
    },
    ui: {
      description: "Fallback video family.",
      durationHint: "Keep clips short until local looping behavior is validated.",
      promptGuide: { title: "Wan2.2 Prompt Guide", path: "/prompt-guides/wan-2-2.md" },
    },
  },
  {
    id: "wan_2_2_t2v_14b",
    name: "Wan2.2 14B (T2V)",
    type: "video",
    capabilities: ["text_to_video"],
    defaults: { duration: 5, fps: 16, resolution: "1280x720", quality: "balanced" },
    limits: {
      durations: [3, 4, 5],
      recommendedMaxDuration: 5,
      fps: [16],
      resolutions: ["832x480", "480x832", "1280x720", "720x1280"],
    },
    ui: {
      description: "Wan2.2 A14B text-to-video (high/low-noise mixture-of-experts).",
      durationHint: "Heavier than 5B — keep clips at 5s or less. Generates at 16fps.",
      promptGuide: { title: "Wan2.2 14B Text-to-Video Prompt Guide", path: "/prompt-guides/wan-2-2-t2v-14b.md" },
    },
  },
  {
    id: "wan_2_2_i2v_14b",
    name: "Wan2.2 14B (I2V)",
    type: "video",
    capabilities: ["image_to_video", "first_last_frame", "extend_clip", "video_bridge"],
    defaults: { duration: 5, fps: 16, resolution: "1280x720", quality: "balanced" },
    limits: {
      durations: [3, 4, 5],
      recommendedMaxDuration: 5,
      fps: [16],
      resolutions: ["832x480", "480x832", "1280x720", "720x1280"],
    },
    ui: {
      description: "Wan2.2 A14B image-to-video (high/low-noise mixture-of-experts).",
      durationHint: "Heavier than 5B — keep clips at 5s or less. Generates at 16fps.",
      promptGuide: { title: "Wan2.2 14B Image-to-Video Prompt Guide", path: "/prompt-guides/wan-2-2-i2v-14b.md" },
    },
  },
];
