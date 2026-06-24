import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  DatasetDoctorDistributions,
  DatasetDoctorReadout,
  ReadinessBadge,
  ReadinessFlagDetails,
} from "./DatasetDoctor.jsx";

// Render helper mirroring the project's createRoot + act convention (no testing-library).
let container;
let root;

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

function mount(node) {
  act(() => root.render(node));
}

function report(overrides = {}) {
  return {
    gate: "needs_attention",
    subScores: { technical: 0.8 },
    counts: { info: 0, warn: 2, fatal: 0 },
    itemCount: 5,
    items: [],
    datasetFlags: [],
    ...overrides,
  };
}

describe("DatasetDoctorReadout gating states", () => {
  it("renders nothing for an unsaved draft (no report, not loading)", () => {
    mount(<DatasetDoctorReadout report={null} loading={false} />);
    expect(container.querySelector(".dataset-doctor")).toBeNull();
  });

  it("shows a pending state while the first fetch is in flight", () => {
    mount(<DatasetDoctorReadout report={null} loading />);
    expect(container.textContent).toContain("Checking dataset");
  });

  it("renders the gate headline, meter, and summary for a warnable set", () => {
    mount(
      <DatasetDoctorReadout
        report={report({
          itemCount: 18,
          items: [
            { itemId: "a", severity: "warn", flags: [{ check: "blur", severity: "warn" }] },
            { itemId: "b", severity: "warn", flags: [{ check: "blur", severity: "warn" }] },
          ],
        })}
      />,
    );
    const node = container.querySelector(".dataset-doctor");
    expect(node.className).toContain("tone-warn");
    expect(container.textContent).toContain("Trainable");
    expect(container.textContent).toContain("18 photos.");
    expect(container.textContent).toContain("2 look blurry");
    // Technical sub-score drives the meter width.
    expect(container.querySelector(".dataset-doctor-meter-fill").style.width).toBe("80%");
  });

  it("renders the Variety meter only once a diversity sub-score is present (sc-6535)", () => {
    // No diversity sub-score yet (pre-embedding-job) → no Variety meter.
    mount(<DatasetDoctorReadout report={report()} />);
    expect(container.querySelector(".dataset-doctor-meter-variety")).toBeNull();

    // With the Tier-1 diversity sub-score → a second, distinct meter at that width.
    mount(<DatasetDoctorReadout report={report({ subScores: { technical: 0.8, diversity: 0.3 } })} />);
    const variety = container.querySelector(".dataset-doctor-meter-variety");
    expect(variety).not.toBeNull();
    expect(variety.querySelector(".dataset-doctor-meter-fill").style.width).toBe("30%");
  });

  it("renders the Caption match meter only once an alignment sub-score is present (sc-6537)", () => {
    mount(<DatasetDoctorReadout report={report()} />);
    expect(container.querySelector(".dataset-doctor-meter-alignment")).toBeNull();

    // alignment is a raw CLIP cosine (~0.15 is a good match), rescaled for display by alignmentPercent.
    mount(<DatasetDoctorReadout report={report({ subScores: { technical: 0.8, alignment: 0.15 } })} />);
    const alignment = container.querySelector(".dataset-doctor-meter-alignment");
    expect(alignment).not.toBeNull();
    expect(alignment.getAttribute("role")).toBe("meter");
    expect(alignment.getAttribute("aria-label")).toBe("Caption match score");
    expect(alignment.getAttribute("aria-valuenow")).toBe("77");
    expect(alignment.getAttribute("title")).toContain("Caption match score");
    expect(alignment.querySelector(".dataset-doctor-meter-fill").style.width).toBe("77%");
  });

  it("surfaces a re-caption action for caption-alignment-flagged items (sc-6537)", () => {
    const onRecaptionFlagged = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [{ check: "caption_alignment", severity: "warn" }] },
            { itemId: "b", flags: [] },
            { itemId: "c", flags: [{ check: "caption_alignment", severity: "warn" }] },
          ],
        })}
        onRecaptionFlagged={onRecaptionFlagged}
      />,
    );
    const button = container.querySelector(".dataset-doctor-actions button");
    expect(button).not.toBeNull();
    expect(button.textContent).toContain("Re-caption 2");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onRecaptionFlagged).toHaveBeenCalledWith(["a", "c"]);
  });

  it("hides the re-caption action with no flagged items or no handler (sc-6537)", () => {
    mount(
      <DatasetDoctorReadout
        report={report({ items: [{ itemId: "a", flags: [{ check: "blur", severity: "warn" }] }] })}
        onRecaptionFlagged={() => {}}
      />,
    );
    expect(container.querySelector(".dataset-doctor-actions")).toBeNull();
    // Even with flagged items, no button without a handler (e.g. the compact readout).
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [{ itemId: "a", flags: [{ check: "caption_alignment", severity: "warn" }] }],
        })}
      />,
    );
    expect(container.querySelector(".dataset-doctor-actions")).toBeNull();
  });

  it("surfaces a drop-duplicates action and removes the planned copies (sc-6539)", () => {
    const onRemoveDuplicates = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          duplicateRemoval: {
            groups: [
              { keep: "a", remove: ["b"] },
              { keep: "c", remove: ["d", "e"] },
            ],
          },
        })}
        onRemoveDuplicates={onRemoveDuplicates}
      />,
    );
    const button = container.querySelector(".dataset-doctor-actions button");
    expect(button).not.toBeNull();
    expect(button.textContent).toContain("Remove 3 duplicates");
    expect(button.textContent).toContain("keeps the sharpest");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onRemoveDuplicates).toHaveBeenCalledWith(["b", "d", "e"]);
  });

  it("hides the drop-duplicates action with no plan or no handler (sc-6539)", () => {
    // A plan present but no handler (e.g. a read-only consumer) → no button.
    mount(
      <DatasetDoctorReadout
        report={report({ duplicateRemoval: { groups: [{ keep: "a", remove: ["b"] }] } })}
      />,
    );
    expect(container.querySelector(".dataset-doctor-actions")).toBeNull();
    // A handler but no removable duplicates → no button.
    mount(
      <DatasetDoctorReadout report={report({})} onRemoveDuplicates={() => {}} />,
    );
    expect(container.querySelector(".dataset-doctor-actions")).toBeNull();
  });

  it("surfaces an upscale action for resolution-flagged items (sc-6539)", () => {
    const onUpscaleLowRes = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [{ check: "resolution", severity: "warn" }] },
            { itemId: "b", flags: [] },
            { itemId: "c", flags: [{ check: "resolution", severity: "warn" }] },
          ],
        })}
        onUpscaleLowRes={onUpscaleLowRes}
      />,
    );
    const button = container.querySelector(".dataset-doctor-actions button");
    expect(button).not.toBeNull();
    expect(button.textContent).toContain("Upscale 2 low-res");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onUpscaleLowRes).toHaveBeenCalledWith(["a", "c"]);
  });

  it("surfaces a smart-crop action for crop_loss-flagged items (sc-6539)", () => {
    const onSmartCrop = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [{ check: "crop_loss", severity: "warn" }] },
            { itemId: "b", flags: [] },
            { itemId: "c", flags: [{ check: "crop_loss", severity: "warn" }] },
          ],
        })}
        onSmartCrop={onSmartCrop}
      />,
    );
    const button = [...container.querySelectorAll(".dataset-doctor-actions button")].find((node) =>
      node.textContent.includes("Smart-crop"),
    );
    expect(button).toBeTruthy();
    expect(button.textContent).toContain("Smart-crop 2 images");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onSmartCrop).toHaveBeenCalledWith(["a", "c"]);
  });

  it("surfaces a strip-metadata action for every item when wired (sc-6539)", () => {
    const onStripExif = vi.fn();
    mount(
      <DatasetDoctorReadout
        report={report({
          items: [
            { itemId: "a", flags: [] },
            { itemId: "b", flags: [] },
          ],
        })}
        onStripExif={onStripExif}
      />,
    );
    const button = [...container.querySelectorAll(".dataset-doctor-actions button")].find((node) =>
      node.textContent.includes("Strip metadata"),
    );
    expect(button).toBeTruthy();
    expect(button.textContent).toContain("Strip metadata from 2 images");
    act(() => button.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onStripExif).toHaveBeenCalled();
  });

  it("renders the blocked headline when the set is untrainable", () => {
    mount(
      <DatasetDoctorReadout
        report={report({
          gate: "blocked",
          itemCount: 3,
          counts: { info: 0, warn: 0, fatal: 1 },
          datasetFlags: [{ check: "count", severity: "fatal", value: 3, threshold: 15 }],
        })}
      />,
    );
    expect(container.querySelector(".dataset-doctor").className).toContain("tone-fatal");
    expect(container.textContent).toContain("Not ready to train");
    expect(container.textContent).toContain("add 12 more");
  });

  it("renders a ready set as ready", () => {
    mount(<DatasetDoctorReadout report={report({ gate: "ready", itemCount: 20, counts: { info: 0, warn: 0, fatal: 0 } })} />);
    expect(container.querySelector(".dataset-doctor").className).toContain("tone-good");
    expect(container.textContent).toContain("Ready to train");
  });
});

describe("ReadinessBadge gating states", () => {
  it("is pending (never good) for an item the report has not covered", () => {
    mount(<ReadinessBadge entry={undefined} />);
    const badge = container.querySelector(".dataset-doctor-badge");
    expect(badge.className).toContain("tone-pending");
    expect(badge.getAttribute("title")).toBe("Not assessed yet");
  });

  it("is pending while loading", () => {
    mount(<ReadinessBadge entry={undefined} loading />);
    expect(container.querySelector(".dataset-doctor-badge").className).toContain("tone-pending");
  });

  it("is good for a covered, clean item", () => {
    mount(<ReadinessBadge entry={{ severity: undefined, flags: [] }} />);
    expect(container.querySelector(".dataset-doctor-badge").className).toContain("tone-good");
  });

  it("surfaces the worst-flag reason for a warned item", () => {
    mount(<ReadinessBadge entry={{ severity: "warn", flags: [{ check: "blur", severity: "warn" }] }} />);
    const badge = container.querySelector(".dataset-doctor-badge");
    expect(badge.className).toContain("tone-warn");
    expect(badge.getAttribute("title")).toContain("blurry");
  });

  it("is fatal for an untrainable item", () => {
    mount(<ReadinessBadge entry={{ severity: "fatal", flags: [{ check: "decode", severity: "fatal" }] }} />);
    expect(container.querySelector(".dataset-doctor-badge").className).toContain("tone-fatal");
  });
});

describe("ReadinessFlagDetails (Advanced raw numbers)", () => {
  it("renders nothing for a clean item", () => {
    mount(<ReadinessFlagDetails entry={{ flags: [] }} />);
    expect(container.querySelector(".dataset-doctor-flags")).toBeNull();
  });

  it("shows the raw value and threshold behind a flag", () => {
    mount(
      <ReadinessFlagDetails
        entry={{ flags: [{ check: "blur", severity: "warn", value: 42, threshold: 100 }] }}
      />,
    );
    expect(container.textContent).toContain("blurry");
    expect(container.textContent).toContain("sharpness");
    expect(container.textContent).toContain("42");
    expect(container.textContent).toContain("threshold 100");
  });

  it("keeps sub-one fractions readable (e.g. clip/crop)", () => {
    mount(
      <ReadinessFlagDetails
        entry={{ flags: [{ check: "exposure", severity: "warn", value: 0.0734, threshold: 0.05 }] }}
      />,
    );
    expect(container.textContent).toContain("0.073");
  });

  it("offers Dismiss on an active finding and calls back to acknowledge it", () => {
    const onToggle = vi.fn();
    mount(
      <ReadinessFlagDetails
        entry={{ flags: [{ check: "blur", severity: "warn", value: 40, threshold: 100 }] }}
        onToggle={onToggle}
      />,
    );
    const button = container.querySelector(".dataset-doctor-flag-toggle");
    expect(button.textContent).toBe("Dismiss");
    act(() => button.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    expect(onToggle).toHaveBeenCalledWith("blur", true);
  });

  it("shows caption alignment reason, cosine metric, and dismiss callback", () => {
    const onToggle = vi.fn();
    mount(
      <ReadinessFlagDetails
        entry={{
          flags: [
            {
              check: "caption_alignment",
              severity: "warn",
              value: 0.12,
              threshold: 0.2,
            },
          ],
        }}
        onToggle={onToggle}
      />,
    );
    expect(container.textContent).toContain("Caption may not match this image");
    expect(container.textContent).toContain("caption-image cosine similarity");
    expect(container.textContent).toContain("0.12");
    expect(container.textContent).toContain("threshold 0.2");
    const button = container.querySelector(".dataset-doctor-flag-toggle");
    act(() => button.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    expect(onToggle).toHaveBeenCalledWith("caption_alignment", true);
  });

  it("shows a dismissed finding struck-through with an Undo affordance", () => {
    const onToggle = vi.fn();
    mount(
      <ReadinessFlagDetails
        entry={{ flags: [{ check: "blur", severity: "warn", value: 40, threshold: 100, acknowledged: true }] }}
        onToggle={onToggle}
      />,
    );
    const li = container.querySelector(".dataset-doctor-flags li");
    expect(li.className).toContain("dismissed");
    const button = container.querySelector(".dataset-doctor-flag-toggle");
    expect(button.textContent).toBe("Undo");
    act(() => button.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    expect(onToggle).toHaveBeenCalledWith("blur", false);
  });

  it("does not offer to dismiss a non-acknowledgeable finding (decode)", () => {
    mount(
      <ReadinessFlagDetails entry={{ flags: [{ check: "decode", severity: "warn" }] }} onToggle={() => {}} />,
    );
    expect(container.querySelector(".dataset-doctor-flag-toggle")).toBeNull();
  });
});

describe("DatasetDoctorDistributions", () => {
  it("renders nothing without distributions", () => {
    mount(<DatasetDoctorDistributions report={{ gate: "ready" }} />);
    expect(container.querySelector(".dataset-doctor-distributions")).toBeNull();
  });

  it("draws a histogram per metric with the threshold marked", () => {
    mount(
      <DatasetDoctorDistributions
        report={{
          distributions: {
            blurVariance: { values: [10, 200, 400, 50], threshold: 100, higherIsBetter: true },
            shadowClip: { values: [0, 0.01, 0.2], threshold: 0.05, higherIsBetter: false },
            highlightClip: { values: [0, 0, 0], threshold: 0.05, higherIsBetter: false },
          },
        }}
      />,
    );
    expect(container.querySelectorAll(".dataset-doctor-histogram").length).toBe(3);
    expect(container.querySelectorAll(".dataset-doctor-histogram-threshold").length).toBe(3);
    expect(container.textContent).toContain("Sharpness");
    expect(container.textContent).toContain("higher is better");
  });
});
