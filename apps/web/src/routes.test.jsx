import { describe, expect, it } from "vitest";
import {
  coerceViewForMode,
  getInitialViewForMode,
  getNavigationSections,
  isViewVisibleInMode,
} from "./routes.jsx";

describe("route visibility by interface mode", () => {
  it("keeps advanced navigation as the default branch", () => {
    const labels = getNavigationSections("advanced").flatMap((section) => section.items.map((item) => item.id));

    expect(labels).toContain("Image");
    expect(labels).toContain("Video");
    expect(labels).toContain("Train");
    expect(getInitialViewForMode("advanced")).toBe("Library");
  });

  it("shows the simple create routes only in simple mode", () => {
    const simpleIds = getNavigationSections("simple").flatMap((section) => section.items.map((item) => item.id));

    expect(simpleIds).toEqual(["MakePicture", "MakeVideo", "SimpleCharacters", "MyCreations", "Queue", "SimpleSettings"]);
    expect(isViewVisibleInMode("MakePicture", "simple")).toBe(true);
    expect(isViewVisibleInMode("MakeVideo", "simple")).toBe(true);
    expect(isViewVisibleInMode("Train", "simple")).toBe(false);
    expect(isViewVisibleInMode("MakePicture", "advanced")).toBe(false);
  });

  it("coerces hidden routes to the active mode's initial view", () => {
    expect(coerceViewForMode("Train", "simple")).toBe("MakePicture");
    expect(coerceViewForMode("MakePicture", "advanced")).toBe("Library");
  });
});
