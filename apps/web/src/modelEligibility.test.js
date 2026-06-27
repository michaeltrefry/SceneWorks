import { describe, expect, it } from "vitest";
import { DEFAULT_MAC_CAPABILITIES } from "./macGating.js";
import {
  angleModelUsable,
  characterModelUsable,
  documentModelUsable,
  downloadOffersFor,
  hasUsableModelFor,
  imageModelUsable,
  poseModelUsable,
  videoModelUsable,
  visionCaptionModelUsable,
} from "./modelEligibility.js";
import { VISION_CAPTION_MODEL_ID } from "./constants.js";

const caps = DEFAULT_MAC_CAPABILITIES; // gating off → Mac blocks are no-ops

describe("modelEligibility predicates", () => {
  it("imageModelUsable matches image models serving a mode, rejects other types", () => {
    expect(imageModelUsable({ type: "image", capabilities: ["text_to_image"] }, caps)).toBe(true);
    expect(imageModelUsable({ type: "image", capabilities: ["edit_image"] }, caps)).toBe(true);
    expect(imageModelUsable({ type: "image", capabilities: [] }, caps)).toBe(false);
    expect(imageModelUsable({ type: "video", capabilities: ["text_to_image"] }, caps)).toBe(false);
  });

  it("videoModelUsable matches video models with a video capability", () => {
    expect(videoModelUsable({ type: "video", capabilities: ["text_to_video"] }, caps)).toBe(true);
    expect(videoModelUsable({ type: "video", capabilities: ["animate_character"] }, caps)).toBe(true);
    expect(videoModelUsable({ type: "video", capabilities: [] }, caps)).toBe(false);
    expect(videoModelUsable({ type: "image", capabilities: ["text_to_video"] }, caps)).toBe(false);
  });

  it("documentModelUsable requires an interleave-capable image model", () => {
    expect(documentModelUsable({ type: "image", capabilities: ["interleave"] }, caps)).toBe(true);
    expect(documentModelUsable({ type: "image", capabilities: ["text_to_image"] }, caps)).toBe(false);
  });

  it("angle/pose predicates read the ui flags", () => {
    expect(angleModelUsable({ ui: { viewAngles: [{ id: "front" }] } }, caps)).toBe(true);
    expect(angleModelUsable({ ui: { viewAngles: [] } }, caps)).toBe(false);
    expect(poseModelUsable({ ui: { poseLibrary: true } }, caps)).toBe(true);
    expect(poseModelUsable({ ui: {} }, caps)).toBe(false);
    expect(characterModelUsable({ ui: { poseLibrary: true } }, caps)).toBe(true);
    expect(characterModelUsable({ ui: { viewAngles: [{ id: "front" }] } }, caps)).toBe(true);
    expect(characterModelUsable({ ui: {} }, caps)).toBe(false);
  });

  it("hasUsableModelFor counts present (installed/incomplete) models, not missing ones", () => {
    const installed = { id: "b", type: "image", capabilities: ["text_to_image"], installState: "installed" };
    const incomplete = { id: "c", type: "image", capabilities: ["text_to_image"], installState: "incomplete" };
    const missing = { id: "a", type: "image", capabilities: ["text_to_image"], installState: "missing" };
    expect(hasUsableModelFor([missing, installed], imageModelUsable, caps)).toBe(true);
    expect(hasUsableModelFor([incomplete], imageModelUsable, caps)).toBe(true);
    expect(hasUsableModelFor([missing], imageModelUsable, caps)).toBe(false);
  });

  // SD3.5 surfacing + eligibility/gating (epic 7841 / sc-7873). The three native MLX variants are
  // text-to-image image models, so they are usable on Image Studio (text_to_image mode) when their
  // macSupport oracle reports supported. Under active Mac gating an unsupported variant (e.g. one
  // without an MLX engine, or any model off-Mac) is blocked from the picker; with gating off the Mac
  // blocks are no-ops so they always surface (Image Studio is the macOnly-aware path).
  it("imageModelUsable surfaces the SD3.5 variants and respects Mac gating", () => {
    const activeCaps = { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: true, platform: "macos" };
    for (const id of ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]) {
      const supported = {
        id,
        type: "image",
        capabilities: ["text_to_image", "style_variations"],
        macSupport: { supported: true, features: {} },
      };
      // Mac-supported native MLX variant → usable on Image Studio under active gating.
      expect(imageModelUsable(supported, activeCaps)).toBe(true);
      // Gating off (non-Mac / observe mode) → Mac block is a no-op, still usable.
      expect(imageModelUsable(supported, caps)).toBe(true);
      // Unsupported (no MLX engine for this variant) → hidden from the picker under active gating.
      const unsupported = { ...supported, macSupport: { supported: false } };
      expect(imageModelUsable(unsupported, activeCaps)).toBe(false);
    }
  });

  // Reference-image vision captioner gate (epic 8102, sc-8110). The captioner is a single pinned
  // utility model; usability = "this IS that model AND it can run here", and macOnly keeps it hidden
  // on non-macOS platforms until the candle path (epic 8103) lands.
  it("visionCaptionModelUsable matches only the captioner model and respects macOnly", () => {
    const captioner = { id: VISION_CAPTION_MODEL_ID, type: "utility", macOnly: true };
    // macOS (or pre-load empty platform) → usable.
    expect(visionCaptionModelUsable(captioner, { ...caps, platform: "macos" })).toBe(true);
    expect(visionCaptionModelUsable(captioner, caps)).toBe(true); // platform "" → no-op pre-load
    // Non-Mac platform with a macOnly model → hidden (epic 8103 flips this).
    expect(visionCaptionModelUsable(captioner, { ...caps, platform: "windows" })).toBe(false);
    expect(visionCaptionModelUsable(captioner, { ...caps, platform: "linux" })).toBe(false);
    // A different model id is never the captioner.
    expect(visionCaptionModelUsable({ id: "some_other_model", macOnly: true }, { ...caps, platform: "macos" })).toBe(false);
    // Active Mac gating with the model's MLX oracle reporting unsupported → blocked.
    const blockedCaps = { ...DEFAULT_MAC_CAPABILITIES, macGatingActive: true, platform: "macos" };
    const unsupported = { ...captioner, macSupport: { supported: false } };
    expect(visionCaptionModelUsable(unsupported, blockedCaps)).toBe(false);
  });

  it("hasUsableModelFor / downloadOffersFor drive the captioner gate (sc-8110)", () => {
    const macCaps = { ...caps, platform: "macos" };
    const installed = { id: VISION_CAPTION_MODEL_ID, type: "utility", macOnly: true, installState: "installed" };
    const missing = { id: VISION_CAPTION_MODEL_ID, type: "utility", macOnly: true, installState: "missing", recommended: true };
    // Present (installed) → screen is "ready".
    expect(hasUsableModelFor([installed], visionCaptionModelUsable, macCaps)).toBe(true);
    // Absent (missing) → not ready, and it surfaces as a recommended-first download offer.
    expect(hasUsableModelFor([missing], visionCaptionModelUsable, macCaps)).toBe(false);
    expect(downloadOffersFor([missing], visionCaptionModelUsable, macCaps).map((m) => m.id)).toEqual([
      VISION_CAPTION_MODEL_ID,
    ]);
    // On Windows the predicate rejects it, so there is no offer (feature stays hidden).
    expect(downloadOffersFor([missing], visionCaptionModelUsable, { ...caps, platform: "windows" })).toEqual([]);
  });

  it("downloadOffersFor prefers recommended, falls back to any eligible, skips installed", () => {
    const models = [
      { id: "rec", type: "image", capabilities: ["text_to_image"], installState: "missing", recommended: true },
      { id: "plain", type: "image", capabilities: ["text_to_image"], installState: "missing" },
      { id: "done", type: "image", capabilities: ["text_to_image"], installState: "installed", recommended: true },
    ];
    expect(downloadOffersFor(models, imageModelUsable, caps).map((m) => m.id)).toEqual(["rec"]);
    // No recommended among eligible → fall back to all eligible (not installed).
    const noRec = models.filter((m) => m.id === "plain");
    expect(downloadOffersFor(noRec, imageModelUsable, caps).map((m) => m.id)).toEqual(["plain"]);
  });
});
