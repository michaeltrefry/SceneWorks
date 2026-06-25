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

// Catalog id of the prompt-refinement / magic-prompt LLM (sc-5605 / sc-6550, builtin.models.jsonc):
// one coherent Anubis-8B serves BOTH the free-text "Refine my prompt" rewrite AND Ideogram 4's
// magic-prompt (plain idea -> structured JSON caption) — the sc-6550 bake-off found the old 3B / plain
// Llama-3.1-8B emit degenerate captions that placeholder. The native worker resolves the model by repo
// string, not this id; the web uses it to look up the catalog entry's install state and to enqueue its
// ModelDownload job when refine / magic-prompt needs it.
export const PROMPT_REFINE_MODEL_ID = "prompt_refine_anubis_8b";

// Catalog id of the dataset-captioning model (sc-5620, builtin.models.jsonc). Same pattern
// as PROMPT_REFINE_MODEL_ID — the native captioner resolves it by repo string; the web uses
// this id to look up install state and offer a download in the caption dialog when missing.
export const JOY_CAPTION_MODEL_ID = "joycaption_beta_one";

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
      // Strict pose tier (sc-2257): on the macOS MLX backend the worker renders
      // the selected library pose to an OpenPose skeleton and conditions the
      // ported Z-Image Fun-Controlnet-Union branch on it (true pose lock). The
      // pose picker gates on this flag alone — no character_image needed.
      poseLibrary: true,
      // Strict ControlNet → expose a pose-lock-strength slider (advanced.controlScale).
      // Best-effort tiers (Qwen/Flux2) have no strength control, so they omit this.
      poseControlScale: true,
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
      // Strict pose tier (sc-2291): torch QwenImageControlNetPipeline + InstantX
      // Qwen-Image-ControlNet-Union (DWPose) — true pose lock, pose-from-prompt.
      // poseLibrary alone gates the pose picker (no character_image needed).
      poseLibrary: true,
      // Strict ControlNet → pose-lock-strength slider (advanced.controlScale).
      poseControlScale: true,
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
    id: "qwen_image_edit_2511",
    name: "Qwen Image Edit (2511)",
    type: "image",
    // character_image (sc-2014): subject consistency via QwenImageEditPlusPipeline;
    // reference goes through `image=` + trueCfgScale. sc-2160 replaces the
    // August/September iterations with the December 2511 release.
    capabilities: ["edit_image", "character_image"],
    ui: {
      description: "December 2025 iteration of Qwen-Image-Edit (Qwen/Qwen-Image-Edit-2511) on QwenImageEditPlusPipeline. Drift mitigation, multi-person consistency, integrated popular LoRAs. Apache-2.0, ungated; 40 steps at trueCfgScale 4.0 / guidanceScale 1.0.",
      promptGuide: { title: "Qwen Image Edit (2511) Prompt Guide", path: "/prompt-guides/qwen-image-edit-2511.md" },
      // Qwen's slider drives trueCfgScale; the IP-Adapter reference-strength
      // slider would be a no-op here. Hide it and surface trueCfgScale instead
      // (sc-2017). Label is "Prompt strength" per sc-2013: identity holds
      // across the range, the knob is a prompt-adherence vs reference tradeoff.
      hideReferenceStrength: true,
      variationStrength: { label: "Prompt strength", default: 4.0, min: 1.0, max: 10.0, step: 0.5 },
    },
  },
  {
    id: "qwen_image_edit_2511_lightning",
    name: "Qwen Image Edit (2511) Lightning",
    type: "image",
    capabilities: ["edit_image", "character_image"],
    ui: {
      description: "4-step distilled Qwen-Image-Edit-2511 (lightx2v Lightning LoRA fused at load). ~10x faster at a small quality trade-off; CFG disabled. In Character Studio's angle set + pose library, the prompt-driven fast tier (sc-2003 spike: mean ArcFace 0.62 across 5 angles, second-best identity after InstantID; pose-library multi-image best-effort, sitting 0.68 / standing 0.45). Caveat: profile prompts reframe to full-body (Qwen's framing-expansion personality).",
      promptGuide: { title: "Qwen Image Edit (2511) Lightning Prompt Guide", path: "/prompt-guides/qwen-image-edit-2511-lightning.md" },
      hideReferenceStrength: true,
      // Distill is trained at cfg 1.0; the lever is exposed only as a narrow
      // safety range (1.0–2.0) for users willing to trade ghosting for stronger
      // prompt adherence. Default stays at 1.0.
      variationStrength: { label: "Prompt strength", default: 1.0, min: 1.0, max: 2.0, step: 0.25 },
      // Multi-backbone angle set (sc-2003): same 11 canonical angles as the
      // InstantID baseline, driven by prompt rather than landmark pack (worker
      // resolves via character_studio_angles.ANGLE_PROMPT_AUGMENTS).
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
      // Best-effort pose library (sc-2256): no pose ControlNet exists for the
      // Qwen edit model, so the worker renders the selected library pose as an
      // OpenPose skeleton and feeds it as a second edit image (image=[reference,
      // skeleton]) on the multi-image Plus pipeline. Strict-tier (InstantID)
      // pose-lock stays the higher-fidelity option; this is the fast best-effort
      // tier validated by the sc-2003 spike (sitting 0.68 / standing 0.45).
      poseLibrary: true,
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
    // character_image (sc-2016): SenseNova's it2i edit path doubles as a
    // wardrobe-preserving Character Studio backbone. Distinct tradeoff vs the
    // face-locked options — sc-2015 spike measured outfit + accessories +
    // tattoos preserved across new scenes, face may drift.
    capabilities: ["text_to_image", "edit_image", "character_image", "vqa", "interleave"],
    limits: { resolutions: ["2048x2048", "2720x1536", "2496x1664", "2368x1760", "1536x2720", "1664x2496", "1760x2368"] },
    ui: {
      description: "Unified multimodal model (NEO-unify, ~16B); native text-to-image and instruction editing with strong text rendering and infographics. In Character Studio, drives a wardrobe-preserving reference flow — outfit + accessories + tattoos + hair color carry through to new scenes, but face geometry may drift. Pick InstantID or PuLID-FLUX for face-locked identity. Heavy (~42GB bf16); CUDA or 96GB+ Apple Silicon.",
      promptGuide: { title: "SenseNova-U1 8B Prompt Guide", path: "/prompt-guides/sensenova-u1-8b.md" },
      // Edit-style variation knob (imageGuidanceScale): higher = closer to the
      // reference, lower = more prompt-driven variation. Drives advanced.imageGuidanceScale.
      variationStrength: { label: "Reference strength", default: 1.5, min: 0.5, max: 4.0, step: 0.1 },
    },
  },
  {
    id: "sensenova_u1_8b_fast",
    name: "SenseNova-U1 8B Fast",
    type: "image",
    // character_image (sc-2016): same wardrobe-preserving reference flow as the
    // base 8B target, on the 8-step distilled variant (~50s/image vs ~4.6 min).
    capabilities: ["text_to_image", "edit_image", "character_image"],
    limits: { resolutions: ["2048x2048", "2720x1536", "2496x1664", "2368x1760", "1536x2720", "1664x2496", "1760x2368"] },
    ui: {
      description: "8-step distilled SenseNova-U1; ~5-6x faster text-to-image, editing, and Character Studio reference (~50s/image on MPS) at a small quality trade-off. Same wardrobe-preserving reference tradeoff as the base 8B (carries outfit + accessories across new scenes; face may drift). Shares the base 8B weights; a ~0.4GB distill LoRA downloads automatically. Distilled editing is experimental — use the base model for max-quality reference work.",
      promptGuide: { title: "SenseNova-U1 8B Fast Prompt Guide", path: "/prompt-guides/sensenova-u1-8b-fast.md" },
      variationStrength: { label: "Reference strength", default: 1.5, min: 0.5, max: 4.0, step: 0.1 },
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
      // FLUX is guidance-distilled; real CFG rides on the parallel trueCfgScale
      // kwarg, distinct from the IP-Adapter reference-strength scalar (sc-2017).
      variationStrength: { label: "Variation", default: 4.0, min: 1.0, max: 10.0, step: 0.5 },
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
    defaults: { resolution: "768x768", steps: 12, guidanceScale: 1.0, sampler: "heun", scheduler: "default" },
    limits: {
      resolutions: ["768x768", "1024x1024", "1280x720", "720x1280"],
      samplers: ["default", "euler", "heun"],
      schedulers: ["default"],
    },
    ui: {
      description: "Chroma1-Flash — fast CFG-baked text-to-image, Apache-2.0 (commercial-safe). FLUX.1-schnell-derived 8.9B + T5-XXL; Heun ~12-step generation, CFG disabled (guidance 1.0, no negative prompt). Large-VRAM GPU.",
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
      // Strict pose tier (sc-2264): official Kolors-ControlNet-Pose + IP-Adapter
      // identity via the vendored Kolors ControlNet pipeline. Real pose-lock.
      poseLibrary: true,
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
    // UI-default seeds for the advanced panel's Guidance / Steps placeholders
    // (sc-3857). These mirror the worker defaults (instantid_adapter.py
    // _guidance_scale / _num_inference_steps) so the form shows the real resolved
    // default; leaving the field blank still defers to the worker. Guidance 3.0
    // (lowered from 5.0 per render tuning — RealVisXL is CFG-sensitive; 3.0 cuts
    // the over-saturated/over-contrast look while holding identity). sampler/
    // scheduler default to "default" (model-native) so behaviour is unchanged
    // until the user picks — e.g. DPM++ SDE + Karras for sharper output.
    defaults: { guidanceScale: 3.0, steps: 30, sampler: "default", scheduler: "default" },
    // SDXL/epsilon sampler menu (sc-3857). The worker registry routes this
    // epsilon pipe to standard solvers; dpmpp_sde + karras == "DPM++ SDE Karras".
    limits: {
      samplers: ["default", "euler", "euler_a", "dpmpp", "dpmpp_sde", "unipc"],
      schedulers: ["default", "karras", "exponential"],
    },
    ui: {
      description: "Identity-preserving character generation — holds a person's face from a single reference image while the prompt drives scene, pose, and wardrobe. RealVisXL_V5.0 (photoreal SDXL, openrail++ commercial-OK) + InstantID ArcFace embedding & landmark ControlNet; faithful likeness with scene freedom (vs IP-Adapter resemblance only). Pick a character with an approved reference, then raise Variations. ~30 steps at guidance 3.0, ~22GB peak.",
      promptGuide: { title: "InstantID (RealVisXL) Prompt Guide", path: "/prompt-guides/instantid-realvisxl.md" },
      // Per-model default negative prompt (sc-3857). Image Studio seeds this into
      // an empty negative box on entering character mode — RealVisXL otherwise ran
      // with NO negative there (the main reason its character output looked worse
      // than Character Studio's). The terms target this model's failure modes:
      // shiny/plastic skin and the over-saturated/over-contrasty look. Editable,
      // and other models can declare their own style-appropriate default.
      defaultNegativePrompt:
        "plastic skin, airbrushed, oversaturated, overexposed, high contrast, cgi, 3d render, cartoon, anime, waxy, deformed, blurry, lowres",
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
    id: "pulid_flux_dev",
    name: "PuLID-FLUX (FLUX.1 [dev])",
    type: "image",
    // Reference-driven only — appears solely in the "With character" picker.
    capabilities: ["character_image"],
    ui: {
      description: "Identity-preserving FLUX character generation — holds a person's face from a single reference image while the prompt drives scene, pose, and wardrobe. FLUX.1-dev + PuLID IDFormer cross-attention; the sc-2012 spike measured 0.8016 ArcFace cosine vs reference at id_weight=1.0 / start-cfg=4 (above InstantID-SDXL no-restore; no face-restoration pass needed). Pick a character with an approved reference, then raise Variations. ~30 steps at guidance 4.0, ~85 GB peak unified memory. License: FLUX.1 [dev] non-commercial (gated). NOT shown in the angle-set or pose-library pickers (sc-2003 spike: identity injection over-anchors the head to the reference's frontal pose; prompt direction is treated as decorative — left/right outputs are visually identical).",
      promptGuide: { title: "PuLID-FLUX Prompt Guide", path: "/prompt-guides/pulid-flux.md" },
      // Identity tuning: reference strength drives idWeight (PuLID's identity-strength
      // analog of InstantID's ip_adapter_scale); identityStructure adds a second slider
      // for timestepToStartCfg (higher = identity injected later in the denoise = more
      // editability but weaker identity; PuLID photoreal recommendation is 4).
      referenceStrengthDefault: 1.0,
      identityStructure: { label: "Identity start step", default: 4, min: 0, max: 8, step: 1 },
    },
  },
  {
    id: "flux2_klein_9b",
    name: "FLUX.2 [klein] 9B",
    type: "image",
    // FLUX.2 [klein] — Apple Silicon MLX-only image-edit backbone. The
    // adapter advertises text_to_image (txt2img) and character_image (reference
    // editing through Flux2KleinEdit + image_paths).
    capabilities: ["text_to_image", "character_image", "style_variations"],
    ui: {
      description: "Black Forest Labs FLUX.2 [klein] 9B — 4-step distilled text-to-image + reference editing, MLX-only (Apple Silicon). Distributed under the FLUX Non-Commercial License (gated). In Character Studio's angle set + pose library, the FLUX.2-aesthetic tier (sc-2003 spike: mean ArcFace 0.52 across 5 angles — third-best identity hold, BUT the only prompt-driven backbone that holds portrait framing at 90° profiles where Qwen and InstantID both reframe). Pose library runs the multi-image trick (skeleton + character) at compact ~22 GB memory.",
      promptGuide: { title: "FLUX.2 [klein] 9B Prompt Guide", path: "/prompt-guides/flux2-klein.md" },
      variationStrength: { label: "Prompt strength", default: 4.0, min: 1.0, max: 10.0, step: 0.5 },
      // Multi-backbone angle set (sc-2003). MlxFlux2Adapter passes the per-
      // angle augmented prompt through the sidecar runner.
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
      // Best-effort pose library (sc-2262): worker renders the selected pose as
      // an OpenPose skeleton and feeds it as a second edit image alongside the
      // reference through the MLX sidecar (Flux2KleinEdit multi-image). Strict
      // InstantID pose-lock stays the higher-fidelity option.
      poseLibrary: true,
    },
  },
  {
    // Stable Diffusion 3.5 Large Turbo (epic 7841 / S4 sc-7873) — the fast default
    // of the SD3.5 family: ADD-distilled few-step (~4), CFG-free 8B MMDiT, native
    // MLX (Apple Silicon). Catalog-driven gating (macOnly + gated) comes from the
    // manifest/macSupport; this fallback entry only seeds the picker + per-variant
    // defaults until the live catalog loads. Recommended (the fast SD3.5 default).
    id: "sd3_5_large_turbo",
    name: "Stable Diffusion 3.5 Large Turbo",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    // Few-step CFG-free: 4 steps, guidance ~1.0 (the negative prompt is inert).
    defaults: { steps: 4, guidanceScale: 1.0, sampler: "euler", scheduler: "default" },
    limits: {
      samplers: ["default", "euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_sde", "uni_pc", "lcm", "ddim"],
      schedulers: ["default", "normal", "simple", "karras", "exponential", "sgm_uniform", "beta", "ddim_uniform"],
    },
    ui: {
      description:
        "Stable Diffusion 3.5 Large Turbo — the few-step (~4), CFG-free distilled SD3.5 flagship: fast iteration at flagship quality on the native MLX engine (Apple Silicon). Best for quick text-to-image; native 1024×1024. Stability AI Community License (gated).",
      promptGuide: { title: "Stable Diffusion 3.5 Prompt Guide", path: "/prompt-guides/sd3-5.md" },
    },
  },
  {
    // Stable Diffusion 3.5 Large (epic 7841 / S4 sc-7873) — the high-fidelity
    // flagship: 8B MMDiT + triple text encoder + true CFG with negative prompts.
    // ~28 steps at guidance 3.5. Native MLX (Apple Silicon), gated.
    id: "sd3_5_large",
    name: "Stable Diffusion 3.5 Large",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    // High-fidelity true-CFG flagship: 28 steps, guidance 3.5 (+negative prompt).
    defaults: { steps: 28, guidanceScale: 3.5, sampler: "euler", scheduler: "default" },
    limits: {
      samplers: ["default", "euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_sde", "uni_pc", "lcm", "ddim"],
      schedulers: ["default", "normal", "simple", "karras", "exponential", "sgm_uniform", "beta", "ddim_uniform"],
    },
    ui: {
      description:
        "Stability AI Stable Diffusion 3.5 Large — the 8B multimodal diffusion transformer (MMDiT) flagship for the highest fidelity, native MLX (Apple Silicon). Triple text encoder for strong prompt adherence and in-image text; true CFG with negative prompts. ~28 steps at guidance 3.5; native 1024×1024. Stability AI Community License (gated).",
      promptGuide: { title: "Stable Diffusion 3.5 Prompt Guide", path: "/prompt-guides/sd3-5.md" },
    },
  },
  {
    // Stable Diffusion 3.5 Medium (epic 7841 / S4 sc-7873) — the smaller-RAM
    // mid-tier: 2.5B MMDiT-X (dual-attention) + the same triple TE + true CFG,
    // renders up to 1440². ~40 steps at guidance 4.5. Native MLX (Apple Silicon),
    // gated; the lightest SD3.5 footprint.
    id: "sd3_5_medium",
    name: "Stable Diffusion 3.5 Medium",
    type: "image",
    capabilities: ["text_to_image", "style_variations"],
    // Smaller-RAM mid-tier: a few more steps + slightly higher guidance than Large.
    defaults: { steps: 40, guidanceScale: 4.5, sampler: "euler", scheduler: "default" },
    limits: {
      samplers: ["default", "euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_sde", "uni_pc", "lcm", "ddim"],
      schedulers: ["default", "normal", "simple", "karras", "exponential", "sgm_uniform", "beta", "ddim_uniform"],
    },
    ui: {
      description:
        "Stability AI Stable Diffusion 3.5 Medium — the 2.5B MMDiT-X (dual-attention) mid-tier with the lightest SD3.5 memory footprint, native MLX (Apple Silicon). Same triple text encoder and true CFG as Large; renders up to 1440×1440. ~40 steps at guidance 4.5. Stability AI Community License (gated).",
      promptGuide: { title: "Stable Diffusion 3.5 Prompt Guide", path: "/prompt-guides/sd3-5.md" },
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
    // Default to 832x480 for a sane out-of-the-box time on the Mac MLX path (sc-4997): measured
    // 5B @ 832x480/121f/20-step/CFG = ~5 min vs ~20 min at 1280x720 (the z48 VAE decode dominates
    // at high res). 1280x720 stays user-selectable via `limits.resolutions` for those who accept it.
    defaults: { duration: 5, fps: 24, resolution: "832x480", quality: "balanced" },
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
  {
    // Wan2.2 VACE-Fun A14B (epic 3456): native-only dual-expert VACE control model. First
    // slice exposes replace_person (the validated VACE path); control-video preprocessing
    // (pose/depth/canny) + extend/bridge are tracked follow-ups (sc-3460).
    id: "wan_2_2_vace_fun_14b",
    name: "Wan2.2 VACE-Fun (A14B)",
    type: "video",
    capabilities: ["replace_person"],
    defaults: { duration: 5, fps: 16, resolution: "832x480", quality: "balanced" },
    limits: {
      durations: [3, 4, 5],
      recommendedMaxDuration: 5,
      fps: [16],
      resolutions: ["832x480", "480x832", "1280x720", "720x1280"],
    },
    ui: {
      description: "Wan2.2 A14B VACE control model (high/low-noise mixture-of-experts) for person replacement and controllable video.",
      durationHint: "Heavy dual-expert control model — keep clips at 5s or less. Generates at 16fps.",
      promptGuide: { title: "Wan2.2 VACE-Fun Prompt Guide", path: "/prompt-guides/wan-2-2-t2v-14b.md" },
    },
  },
];
