import { describe, expect, it } from "vitest";
import { parseResolution, pickClosestResolution } from "./resolutionMatch.js";

describe("parseResolution", () => {
  it("parses WxH strings", () => {
    expect(parseResolution("1024x576")).toEqual({ width: 1024, height: 576 });
  });

  it("returns null for malformed input", () => {
    expect(parseResolution("1024")).toBeNull();
    expect(parseResolution("axb")).toBeNull();
    expect(parseResolution("")).toBeNull();
    expect(parseResolution(null)).toBeNull();
    expect(parseResolution(undefined)).toBeNull();
  });

  it("rejects non-positive dimensions", () => {
    expect(parseResolution("0x720")).toBeNull();
  });
});

describe("pickClosestResolution", () => {
  const ltxOptions = ["768x512", "512x768", "640x640", "1024x1024", "1280x720", "720x1280"];

  it("returns null when options or source dims are missing/invalid", () => {
    expect(pickClosestResolution(1024, 1024, [])).toBeNull();
    expect(pickClosestResolution(1024, 1024, null)).toBeNull();
    expect(pickClosestResolution(0, 1024, ltxOptions)).toBeNull();
    expect(pickClosestResolution(1024, 0, ltxOptions)).toBeNull();
    expect(pickClosestResolution(NaN, 1024, ltxOptions)).toBeNull();
  });

  it("picks the square option for a square source", () => {
    expect(pickClosestResolution(1500, 1500, ltxOptions)).toBe("1024x1024");
    expect(pickClosestResolution(500, 500, ltxOptions)).toBe("640x640");
  });

  it("picks landscape 16:9 for a 1920x1080 source", () => {
    expect(pickClosestResolution(1920, 1080, ltxOptions)).toBe("1280x720");
  });

  it("picks portrait 9:16 for a 1080x1920 source", () => {
    expect(pickClosestResolution(1080, 1920, ltxOptions)).toBe("720x1280");
  });

  it("picks landscape 3:2 for a 3000x2000 source", () => {
    expect(pickClosestResolution(3000, 2000, ltxOptions)).toBe("768x512");
  });

  it("picks portrait 2:3 for a 2000x3000 source", () => {
    expect(pickClosestResolution(2000, 3000, ltxOptions)).toBe("512x768");
  });

  it("skips malformed options without throwing", () => {
    expect(pickClosestResolution(1500, 1500, ["bad", "1024x1024"])).toBe("1024x1024");
  });

  it("returns null when every option is malformed", () => {
    expect(pickClosestResolution(1500, 1500, ["bad", "alsobad"])).toBeNull();
  });

  it("returns the sole option when only one is available", () => {
    expect(pickClosestResolution(1920, 1080, ["1024x1024"])).toBe("1024x1024");
  });

  // The Image Studio reference-image auto-preset (sc-8109): the resolution picker is
  // driven by selectedModel.limits.resolutions, falling back to these defaults. A
  // captioned reference must render at a matching aspect or its frame-normalized
  // (0–1000) bboxes come out wrong-shaped.
  describe("Image Studio reference-image presets (sc-8109)", () => {
    // The DEFAULT_RESOLUTION_OPTIONS fallback used when a model declares no
    // limits.resolutions. Note it carries two square buckets, so a mildly-off-square
    // reference (3:4 / 4:3) correctly snaps to a square — square is genuinely the
    // closest aspect in log-space — not to the 16:9 / 9:16 extremes.
    const imageOptions = ["768x768", "1024x1024", "1280x720", "720x1280"];

    it("snaps a strongly-portrait (9:16) reference to the portrait option", () => {
      expect(pickClosestResolution(1080, 1920, imageOptions)).toBe("720x1280");
    });

    it("snaps a strongly-landscape (16:9) reference to the landscape option", () => {
      expect(pickClosestResolution(1920, 1080, imageOptions)).toBe("1280x720");
    });

    it("snaps a mildly-portrait (3:4) reference to the nearer square bucket", () => {
      expect(pickClosestResolution(1200, 1600, imageOptions)).toBe("1024x1024");
    });

    it("picks the larger square for a large square reference (pixel tiebreak)", () => {
      expect(pickClosestResolution(2048, 2048, imageOptions)).toBe("1024x1024");
    });

    it("prefers the portrait bucket over the landscape one for a tall reference", () => {
      // A 2:3 portrait reference is closer to 720x1280 (9:16) than to either square
      // or the 16:9 landscape — it must never snap to a landscape bucket.
      const singleSquare = ["768x768", "1280x720", "720x1280"];
      expect(pickClosestResolution(2000, 3000, singleSquare)).toBe("720x1280");
    });
  });

  // log-ratio symmetry: a W×H source and its H×W mirror must each snap to the
  // mirror-image option. Using a symmetric option set (the same aspects flipped)
  // guarantees the picker treats 2:1 and 1:2 as equidistant from square, not biased.
  describe("log-ratio symmetry", () => {
    const symmetricOptions = ["1000x500", "500x1000", "1000x1000"];

    it("maps a 2:1 source and its 1:2 mirror to mirror-image options", () => {
      expect(pickClosestResolution(2000, 1000, symmetricOptions)).toBe("1000x500");
      expect(pickClosestResolution(1000, 2000, symmetricOptions)).toBe("500x1000");
    });

    it("treats 4:1 and 1:4 sources symmetrically (neither biased toward square)", () => {
      expect(pickClosestResolution(4000, 1000, symmetricOptions)).toBe("1000x500");
      expect(pickClosestResolution(1000, 4000, symmetricOptions)).toBe("500x1000");
    });
  });

  it("prefers an exact aspect match over a closer-pixel-count mismatch", () => {
    // 1500x1000 is exactly 3:2; 1024x1024 (square) has a nearer pixel count but
    // wrong aspect — aspect distance must dominate.
    expect(pickClosestResolution(1500, 1000, ["1024x1024", "1536x1024"])).toBe("1536x1024");
  });

  it("breaks an aspect tie by closest total pixels", () => {
    // Both options are square (zero aspect distance); the larger source prefers the
    // larger square. Asserts the documented pixel-distance tiebreak.
    expect(pickClosestResolution(4000, 4000, ["512x512", "2048x2048"])).toBe("2048x2048");
    expect(pickClosestResolution(400, 400, ["512x512", "2048x2048"])).toBe("512x512");
  });
});
