export const navItems = ["Library", "Image", "Video", "Presets", "Models", "Characters", "Editor", "Queue"];
export const terminalStatuses = new Set(["completed", "failed", "canceled", "interrupted"]);
export const actionStatuses = new Set(["failed", "canceled", "interrupted", "completed"]);
export const fallbackModels = [
  {
    id: "z_image_turbo",
    name: "Z-Image-Turbo",
    type: "image",
    capabilities: ["text_to_image", "style_variations", "character_image"],
    ui: { description: "Fast local text-to-image target." },
  },
  {
    id: "qwen_image",
    name: "Qwen Image",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: { description: "Qwen text-to-image target." },
  },
  {
    id: "z_image_edit",
    name: "Z-Image-Edit",
    type: "image",
    capabilities: ["edit_image"],
    ui: { description: "Image edit target." },
  },
  {
    id: "qwen_image_edit",
    name: "Qwen Image Edit",
    type: "image",
    capabilities: ["edit_image"],
    ui: { description: "Qwen image edit target." },
  },
  {
    id: "lens_turbo",
    name: "Lens-Turbo",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    ui: { description: "Microsoft Lens distilled 4-step text-to-image; strong text rendering, large-VRAM GPU." },
  },
  {
    id: "sensenova_u1_8b",
    name: "SenseNova-U1 8B",
    type: "image",
    capabilities: ["text_to_image", "edit_image"],
    limits: { resolutions: ["2048x2048", "2720x1536", "2496x1664", "2368x1760", "1536x2720", "1664x2496", "1760x2368"] },
    ui: { description: "Unified multimodal model (NEO-unify, ~16B); native text-to-image and instruction editing with strong text rendering and infographics. Heavy (~42GB bf16); CUDA or 96GB+ Apple Silicon." },
  },
  {
    id: "sensenova_u1_8b_fast",
    name: "SenseNova-U1 8B Fast",
    type: "image",
    capabilities: ["text_to_image", "edit_image"],
    limits: { resolutions: ["2048x2048", "2720x1536", "2496x1664", "2368x1760", "1536x2720", "1664x2496", "1760x2368"] },
    ui: { description: "8-step distilled SenseNova-U1; ~5-6x faster text-to-image and editing (~50s/image on MPS) at a small quality trade-off. Shares the base 8B weights; a ~0.4GB distill LoRA downloads automatically. Distilled editing is experimental — use the base model for max-quality edits." },
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
    },
  },
];
