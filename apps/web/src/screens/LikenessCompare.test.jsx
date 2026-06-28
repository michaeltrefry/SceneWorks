import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LikenessCompare } from "./characterPanels.jsx";

// On-demand "compare image to another" likeness UI (epic 4406, sc-4415). Covers the compare states:
// scored (a banded percentage), N/A (the honest no-frontal-face chip, not a low number), error
// (inline failure), and absent (no approved Reference Asset → nothing to compare against).

function click(el) {
  el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

const APPROVED_REFS = [
  { assetId: "ref_a", asset: { id: "ref_a", displayName: "Hero front" } },
  { assetId: "ref_b", asset: { id: "ref_b", displayName: "Hero profile" } },
];
const CANDIDATE = { id: "asset_candidate", type: "image", displayName: "Test render" };

describe("LikenessCompare", () => {
  let container;
  let root;

  async function clickAndSettle(el) {
    await act(async () => {
      click(el);
      await new Promise((r) => setTimeout(r, 0));
    });
  }

  function render(node) {
    act(() => {
      root.render(node);
    });
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("renders nothing when the character has no approved reference (no source identity)", () => {
    render(
      <LikenessCompare
        approvedReferences={[]}
        candidateAsset={CANDIDATE}
        compareFaceLikeness={vi.fn()}
        projectId="proj_1"
      />,
    );
    expect(container.textContent).toBe("");
  });

  it("scores a candidate and renders a banded percentage from the shared scorer result", async () => {
    const compareFaceLikeness = vi.fn().mockResolvedValue({
      score: 0.91,
      detected: true,
      method: "arcface_antelopev2",
      sourceRef: "ref_a",
    });
    render(
      <LikenessCompare
        approvedReferences={APPROVED_REFS}
        candidateAsset={CANDIDATE}
        compareFaceLikeness={compareFaceLikeness}
        projectId="proj_1"
      />,
    );
    // Open the control, then run the compare.
    await clickAndSettle(container.querySelector(".likeness-compare-toggle"));
    const compareButton = [...container.querySelectorAll("button")].find(
      (b) => b.textContent === "Compare",
    );
    await clickAndSettle(compareButton);

    // The runner is called with the selected source + the candidate + the project.
    expect(compareFaceLikeness).toHaveBeenCalledWith({
      sourceAssetId: "ref_a",
      candidateAssetId: "asset_candidate",
      projectId: "proj_1",
    });
    // A strong score renders the percentage badge (91%), not an N/A chip.
    expect(container.textContent).toContain("91%");
  });

  it("renders the honest N/A chip for a no-face candidate, not a low number", async () => {
    const compareFaceLikeness = vi.fn().mockResolvedValue({
      score: null,
      detected: false,
      method: "arcface_antelopev2",
      sourceRef: "ref_a",
      reason: "no_face",
    });
    render(
      <LikenessCompare
        approvedReferences={APPROVED_REFS}
        candidateAsset={CANDIDATE}
        compareFaceLikeness={compareFaceLikeness}
        projectId="proj_1"
      />,
    );
    await clickAndSettle(container.querySelector(".likeness-compare-toggle"));
    const compareButton = [...container.querySelectorAll("button")].find(
      (b) => b.textContent === "Compare",
    );
    await clickAndSettle(compareButton);

    // The N/A badge renders a neutral em-dash chip + the "No frontal face" copy, never a 0% number.
    expect(container.textContent).toContain("—");
    expect(container.textContent).not.toContain("0%");
  });

  it("surfaces an inline error when the compare fails, without crashing", async () => {
    const compareFaceLikeness = vi.fn().mockRejectedValue(new Error("Worker offline"));
    render(
      <LikenessCompare
        approvedReferences={APPROVED_REFS}
        candidateAsset={CANDIDATE}
        compareFaceLikeness={compareFaceLikeness}
        projectId="proj_1"
      />,
    );
    await clickAndSettle(container.querySelector(".likeness-compare-toggle"));
    const compareButton = [...container.querySelectorAll("button")].find(
      (b) => b.textContent === "Compare",
    );
    await clickAndSettle(compareButton);

    const alert = container.querySelector("[role='alert']");
    expect(alert).not.toBeNull();
    expect(alert.textContent).toContain("Worker offline");
  });
});
