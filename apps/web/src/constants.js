export const navItems = ["Library", "Image", "Video", "Models", "Characters", "Editor", "Queue"];
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
    id: "z_image_edit",
    name: "Z-Image-Edit",
    type: "image",
    capabilities: ["edit_image"],
    ui: { description: "Image edit target." },
  },
  {
    id: "ltx_2_3",
    name: "LTX-2.3",
    type: "video",
    capabilities: ["image_to_video", "text_to_video", "first_last_frame"],
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
    capabilities: ["image_to_video", "text_to_video", "first_last_frame", "replace_person"],
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
];
