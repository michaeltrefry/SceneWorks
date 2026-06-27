import React, { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import StructuredPromptBuilder from "./StructuredPromptBuilder.jsx";
import { emptyCaption, serializeCaption, validateCaption } from "../ideogramCaption.js";

// Native-setter trick so React's onChange fires for controlled inputs/textareas.
function setValue(el, value) {
  const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value").set.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

function click(el) {
  el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

describe("StructuredPromptBuilder", () => {
  let container;
  let root;
  let snap;

  // Stateful wrapper so we can drive the controlled component and read back the
  // caption it builds.
  function Harness({
    initialMode = "form",
    initialCaption,
    initialPlain = "",
    onMagicExpand,
    magicModelMissing = false,
    onDownloadMagicModel,
    onImageCaption,
    referenceAssets = [],
    projectId = "",
    visionCaptionReady = true,
    visionCaptionOffers = [],
    visionCaptionDownloadJobs = [],
    onDownloadModel,
    onOpenModels,
    onOpenQueue,
    onCancelJob,
    onReferenceImageLoaded,
  }) {
    const [caption, setCaption] = useState(initialCaption ?? emptyCaption());
    const [mode, setMode] = useState(initialMode);
    const [plain, setPlain] = useState(initialPlain);
    const validation = validateCaption(caption, { plainText: plain });
    snap = { caption, mode, plain, validation };
    return (
      <StructuredPromptBuilder
        caption={caption}
        onCaptionChange={setCaption}
        validation={validation}
        mode={mode}
        onModeChange={setMode}
        plainText={plain}
        onPlainTextChange={setPlain}
        onMagicExpand={onMagicExpand}
        magicModelMissing={magicModelMissing}
        onDownloadMagicModel={onDownloadMagicModel}
        onImageCaption={onImageCaption}
        referenceAssets={referenceAssets}
        projectId={projectId}
        visionCaptionReady={visionCaptionReady}
        visionCaptionOffers={visionCaptionOffers}
        visionCaptionDownloadJobs={visionCaptionDownloadJobs}
        onDownloadModel={onDownloadModel}
        onOpenModels={onOpenModels}
        onOpenQueue={onOpenQueue}
        onCancelJob={onCancelJob}
        onReferenceImageLoaded={onReferenceImageLoaded}
      />
    );
  }

  // Click + let an async handler's promise chain + state updates settle.
  async function clickAndSettle(el) {
    await act(async () => {
      click(el);
      await new Promise((r) => setTimeout(r, 0));
    });
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    snap = null;
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const byPlaceholder = (prefix) => container.querySelector(`[placeholder^="${prefix}"]`);
  const buttonByText = (text) =>
    [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === text);

  async function mount(props = {}) {
    await act(async () => root.render(<Harness {...props} />));
  }

  it("builds a valid, key-ordered caption from the form fields", async () => {
    await mount();
    await act(async () => setValue(byPlaceholder("One sentence"), "A red fox in the snow"));
    await act(async () => setValue(byPlaceholder("The scene behind"), "A snowy pine forest"));
    await act(async () => click(buttonByText("+ Object")));
    await act(async () => setValue(byPlaceholder("Material, pose"), "a red fox sitting upright"));

    expect(snap.validation.ok).toBe(true);
    expect(serializeCaption(snap.caption)).toBe(
      '{"high_level_description": "A red fox in the snow", "compositional_deconstruction": {"background": "A snowy pine forest", "elements": [{"type": "obj", "desc": "a red fox sitting upright"}]}}',
    );
  });

  it("orders text-element keys (type, bbox, text, desc) when toggled to Text with a box", async () => {
    await mount();
    await act(async () => setValue(byPlaceholder("The scene behind"), "bg"));
    await act(async () => click(buttonByText("+ Text")));
    await act(async () => setValue(byPlaceholder("The exact characters"), "HELLO"));
    await act(async () => setValue(byPlaceholder("Material, pose"), "bold serif headline"));
    await act(async () => click(buttonByText("+ Bounding box")));

    expect(snap.validation.ok).toBe(true);
    expect(serializeCaption(snap.caption)).toBe(
      '{"compositional_deconstruction": {"background": "bg", "elements": [{"type": "text", "bbox": [0, 0, 1000, 1000], "text": "HELLO", "desc": "bold serif headline"}]}}',
    );
  });

  it("adds normalized swatches to an element via the palette editor", async () => {
    await mount();
    await act(async () => setValue(byPlaceholder("The scene behind"), "bg"));
    await act(async () => click(buttonByText("+ Object")));
    const hex = container.querySelector('input[aria-label="Hex color"]');
    await act(async () => setValue(hex, "#ff0000"));
    await act(async () => click(buttonByText("Add")));
    await act(async () => setValue(hex, "#00ff00"));
    await act(async () => click(buttonByText("Add")));

    const el = snap.caption.compositional_deconstruction.elements[0];
    expect(el.color_palette).toEqual(["#FF0000", "#00FF00"]);
  });

  it("round-trips an overall/document palette into the caption JSON (sc-5996)", async () => {
    await mount();
    await act(async () => setValue(byPlaceholder("The scene behind"), "a calm studio"));
    // Enable style so the document-level palette is available (the schema only
    // allows an overall palette inside a style block with a discriminator).
    await act(async () => click(container.querySelector('.structured-checkline input[type="checkbox"]')));
    await act(async () => setValue(byPlaceholder("telephoto"), "studio strobe, eye-level"));
    const hex = container.querySelector('input[aria-label="Hex color"]');
    await act(async () => setValue(hex, "#0A2540"));
    await act(async () => click(buttonByText("Add")));

    expect(snap.validation.ok).toBe(true);
    expect(snap.caption.style_description.color_palette).toEqual(["#0A2540"]);
    expect(serializeCaption(snap.caption)).toContain('"color_palette": ["#0A2540"]');
  });

  it("shows the live ordered-JSON preview and applies raw-JSON edits", async () => {
    await mount({ initialMode: "json" });
    const editor = container.querySelector('textarea[aria-label="JSON caption"]');
    expect(editor).toBeTruthy();

    const edited = '{"compositional_deconstruction": {"background": "new bg", "elements": []}}';
    await act(async () => setValue(editor, edited));
    expect(snap.caption.compositional_deconstruction.background).toBe("new bg");

    // Invalid JSON surfaces an error and preserves the last valid caption.
    await act(async () => setValue(editor, "{not json"));
    expect(container.querySelector(".structured-error")).toBeTruthy();
    expect(snap.caption.compositional_deconstruction.background).toBe("new bg");
  });

  it("renders the plain-text fallback and forwards edits", async () => {
    await mount({ initialMode: "plain" });
    const plain = container.querySelector('textarea[aria-label="Plain prompt"]');
    expect(plain).toBeTruthy();
    await act(async () => setValue(plain, "a fox in the snow"));
    expect(snap.plain).toBe("a fox in the snow");
    // No JSON preview in plain mode.
    expect(container.querySelector('[aria-label="Caption preview"]')).toBeFalsy();
  });

  it("switches modes via the segmented control", async () => {
    await mount();
    const jsonTab = [...container.querySelectorAll(".structured-mode button")].find(
      (b) => b.textContent.trim() === "JSON",
    );
    await act(async () => click(jsonTab));
    expect(snap.mode).toBe("json");
  });

  it("expands a plain idea into the editable builder via magic-prompt (sc-5997)", async () => {
    const expanded = {
      high_level_description: "a red fox",
      compositional_deconstruction: { background: "snow", elements: [{ type: "obj", desc: "a red fox" }] },
    };
    const onMagicExpand = vi.fn(async () => expanded);
    await mount({ initialMode: "plain", initialPlain: "a red fox in the snow", onMagicExpand });
    await clickAndSettle(buttonByText("✨ Expand to caption"));

    expect(onMagicExpand).toHaveBeenCalledWith("a red fox in the snow");
    expect(snap.mode).toBe("form"); // dropped into the builder
    expect(snap.caption).toEqual(expanded);
  });

  it("surfaces a magic-prompt error and stays on plain text", async () => {
    const onMagicExpand = vi.fn(async () => {
      throw new Error("expansion blew up");
    });
    await mount({ initialMode: "plain", initialPlain: "an idea", onMagicExpand });
    await clickAndSettle(buttonByText("✨ Expand to caption"));

    expect(container.querySelector(".structured-error")?.textContent).toContain("expansion blew up");
    expect(snap.mode).toBe("plain");
  });

  it("offers a model download when the magic model is missing", async () => {
    const onMagicExpand = vi.fn(async () => {
      throw new Error("snapshot is not cached");
    });
    const onDownloadMagicModel = vi.fn(async () => ({ id: "job1" }));
    await mount({
      initialMode: "plain",
      initialPlain: "an idea",
      onMagicExpand,
      magicModelMissing: true,
      onDownloadMagicModel,
    });
    await clickAndSettle(buttonByText("✨ Expand to caption"));

    const download = buttonByText("Download prompt-refiner model");
    expect(download).toBeTruthy();
    await clickAndSettle(download);
    expect(onDownloadMagicModel).toHaveBeenCalled();
  });

  it("hides the magic-prompt button when no expander is provided", async () => {
    await mount({ initialMode: "plain", initialPlain: "an idea" });
    expect(buttonByText("✨ Expand to caption")).toBeFalsy();
  });

  // ----- reference-image → JSON caption (epic 8102, sc-8108) -----

  const refAsset = {
    id: "ref-1",
    type: "image",
    projectId: "proj-1",
    file: { path: "uploads/ref.png", mimeType: "image/png" },
  };

  // Pick a reference image through the asset picker modal so the caption button enables.
  async function selectReference() {
    await clickAndSettle(buttonByText("Select reference image"));
    const card = container.querySelector(".asset-picker-card");
    await clickAndSettle(card);
    await clickAndSettle(buttonByText("Use Selection"));
  }

  it("captions a picked reference image into the editable builder (sc-8108)", async () => {
    const captioned = {
      high_level_description: "a red fox",
      compositional_deconstruction: {
        background: "snow",
        elements: [{ type: "obj", bbox: [100, 100, 500, 500], desc: "a red fox" }],
      },
    };
    const onImageCaption = vi.fn(async () => captioned);
    await mount({ initialMode: "plain", onImageCaption, referenceAssets: [refAsset], projectId: "proj-1" });
    await selectReference();
    await clickAndSettle(buttonByText("✨ Generate JSON from image"));

    expect(onImageCaption).toHaveBeenCalledWith("ref-1");
    expect(snap.mode).toBe("form"); // dropped into the builder
    // bboxes are KEPT (parseVisionCaption strips aspect_ratio only).
    expect(snap.caption).toEqual(captioned);
  });

  it("keeps the caption button disabled until a reference is selected (sc-8108)", async () => {
    const onImageCaption = vi.fn(async () => ({}));
    await mount({ initialMode: "plain", onImageCaption, referenceAssets: [refAsset], projectId: "proj-1" });
    expect(buttonByText("✨ Generate JSON from image").disabled).toBe(true);
    await selectReference();
    expect(buttonByText("✨ Generate JSON from image").disabled).toBe(false);
  });

  it("surfaces an image-caption error and stays on plain text (sc-8108)", async () => {
    const onImageCaption = vi.fn(async () => {
      throw new Error("captioning blew up");
    });
    await mount({ initialMode: "plain", onImageCaption, referenceAssets: [refAsset], projectId: "proj-1" });
    await selectReference();
    await clickAndSettle(buttonByText("✨ Generate JSON from image"));

    expect(container.querySelector(".structured-error")?.textContent).toContain("captioning blew up");
    expect(snap.mode).toBe("plain");
  });

  it("shows the gate download offer (not the button) when the captioner is missing (sc-8110)", async () => {
    // sc-8110: when the vision captioner isn't installed the section is gated PROACTIVELY through the
    // shared ModelAvailabilityGate — the reference picker + caption button never render, and a download
    // offer is shown instead of a button that would only fail on click.
    const onImageCaption = vi.fn(async () => ({}));
    const onDownloadModel = vi.fn();
    const offer = { id: "vision_caption_qwen3vl_8b", name: "Vision Captioner", downloadSizeLabel: "18 GB" };
    await mount({
      initialMode: "plain",
      onImageCaption,
      referenceAssets: [refAsset],
      projectId: "proj-1",
      visionCaptionReady: false,
      visionCaptionOffers: [offer],
      onDownloadModel,
    });

    // The live-feature controls are hidden behind the gate.
    expect(buttonByText("✨ Generate JSON from image")).toBeFalsy();
    expect(buttonByText("Select reference image")).toBeFalsy();
    // The gate renders its download offer for the captioner.
    expect(container.querySelector(".model-availability-gate")).toBeTruthy();
    const download = buttonByText("Download");
    expect(download).toBeTruthy();
    await clickAndSettle(download);
    expect(onDownloadModel).toHaveBeenCalledWith(offer);
  });

  it("shows the live reference flow (no gate) when the captioner is present (sc-8110)", async () => {
    const onImageCaption = vi.fn(async () => ({}));
    await mount({
      initialMode: "plain",
      onImageCaption,
      referenceAssets: [refAsset],
      projectId: "proj-1",
      visionCaptionReady: true,
    });
    // Ready → the live picker + button render, and the gate is NOT shown.
    expect(container.querySelector(".model-availability-gate")).toBeFalsy();
    expect(buttonByText("Select reference image")).toBeTruthy();
    expect(buttonByText("✨ Generate JSON from image")).toBeTruthy();
  });

  it("hides the reference-image flow when no captioner is wired (gating, sc-8108)", async () => {
    await mount({ initialMode: "plain", initialPlain: "an idea" });
    expect(buttonByText("✨ Generate JSON from image")).toBeFalsy();
    expect(container.querySelector(".structured-reference")).toBeFalsy();
  });

  it("reports blocking validation errors in the preview (out-of-range bbox)", async () => {
    const bad = {
      compositional_deconstruction: {
        background: "bg",
        elements: [{ type: "obj", bbox: [0, 0, 5000, 10], desc: "d" }],
      },
    };
    await mount({ initialCaption: bad });
    expect(snap.validation.ok).toBe(false);
    expect(container.querySelector(".structured-issues-error")).toBeTruthy();
  });

  it("shows the read-only caption preview in form mode (sc-8114)", async () => {
    await mount({ initialMode: "form" });
    expect(container.querySelector('[aria-label="Caption preview"]')).toBeTruthy();
  });

  it("hides the duplicate read-only caption preview on the JSON tab (sc-8114)", async () => {
    await mount({ initialMode: "json" });
    // The JSON textarea already shows the canonical JSON, so the duplicate
    // read-only <pre> preview must NOT render on the JSON tab.
    expect(container.querySelector('[aria-label="Caption preview"]')).toBeFalsy();
    // The editable JSON textarea is still present.
    expect(container.querySelector('textarea[aria-label="JSON caption"]')).toBeTruthy();
  });

  it("hides the read-only caption preview in plain mode (sc-8114)", async () => {
    await mount({ initialMode: "plain" });
    expect(container.querySelector('[aria-label="Caption preview"]')).toBeFalsy();
  });

  it("still surfaces schema validation errors on the JSON tab without the preview (sc-8114)", async () => {
    const bad = {
      compositional_deconstruction: {
        background: "bg",
        elements: [{ type: "obj", bbox: [0, 0, 5000, 10], desc: "d" }],
      },
    };
    await mount({ initialMode: "json", initialCaption: bad });
    // No duplicate preview pane...
    expect(container.querySelector('[aria-label="Caption preview"]')).toBeFalsy();
    // ...but the schema validation errors are still visible on the JSON tab.
    expect(snap.validation.ok).toBe(false);
    expect(container.querySelector(".structured-issues-error")).toBeTruthy();
  });

  // ----- accordion / collapsible elements (sc-8115) -----

  // Helpers scoped to the element list. An expanded row renders the editable
  // <textarea>; a collapsed row renders only the one-line summary button.
  const elementRows = () => [...container.querySelectorAll(".structured-element")];
  const expandedRows = () => [...container.querySelectorAll(".structured-element.expanded")];
  const summaryButtons = () => [...container.querySelectorAll(".structured-element-summary")];

  const captionWithElements = (els) => ({
    compositional_deconstruction: { background: "bg", elements: els },
  });

  it("collapses non-edited elements to a one-line summary, exactly one expanded (sc-8115)", async () => {
    await mount({
      initialCaption: captionWithElements([
        { type: "obj", desc: "a red fox" },
        { type: "text", text: "HELLO", desc: "bold headline" },
        { type: "obj", bbox: [0, 0, 1000, 1000], desc: "a snowy hill", color_palette: ["#FFFFFF"] },
      ]),
    });

    // Three rows render, but only one is expanded at a time.
    expect(elementRows()).toHaveLength(3);
    expect(expandedRows()).toHaveLength(1);
    // The other two collapse to summaries.
    expect(summaryButtons()).toHaveLength(2);

    // Summaries carry the chosen label: `desc` for objects, `text` for text
    // elements (the first two rows are collapsed by default — last is expanded).
    const summaryText = summaryButtons()
      .map((b) => b.querySelector(".structured-element-summary-text").textContent)
      .join("|");
    expect(summaryText).toContain("a red fox");
    expect(summaryText).toContain("HELLO");

    // The bbox/palette indicators show on the collapsed row that has them.
    // (Row 3 is expanded by default, so collapse it to inspect its summary.)
    click(summaryButtons()[0]); // expand row 1, collapsing row 3
    await act(async () => {});
    const row3Summary = summaryButtons().find((b) =>
      b.querySelector(".structured-element-summary-text").textContent.includes("a snowy hill"),
    );
    expect(row3Summary.querySelectorAll(".structured-element-chip")).toHaveLength(2); // box + palette
  });

  it("expands a clicked collapsed row and collapses the others (single-expand) (sc-8115)", async () => {
    await mount({
      initialCaption: captionWithElements([
        { type: "obj", desc: "first" },
        { type: "obj", desc: "second" },
        { type: "obj", desc: "third" },
      ]),
    });
    // Default expanded is the last row.
    expect(expandedRows()[0].querySelector("textarea").value).toBe("third");

    // Click the summary for the "first" row.
    const firstSummary = summaryButtons().find((b) =>
      b.querySelector(".structured-element-summary-text").textContent.includes("first"),
    );
    await act(async () => click(firstSummary));

    // Still exactly one expanded, and it is now "first".
    expect(expandedRows()).toHaveLength(1);
    expect(expandedRows()[0].querySelector("textarea").value).toBe("first");
  });

  it("expands the newly added element, collapsing whatever was open (sc-8115)", async () => {
    await mount({
      initialCaption: captionWithElements([{ type: "obj", desc: "existing" }]),
    });
    expect(expandedRows()[0].querySelector("textarea").value).toBe("existing");

    await act(async () => click(buttonByText("+ Object")));

    // The new (empty) element is the expanded one; the prior row collapsed.
    expect(elementRows()).toHaveLength(2);
    expect(expandedRows()).toHaveLength(1);
    expect(expandedRows()[0].querySelector("textarea").value).toBe("");
    // Edits land on the new element, confirming it is the editable one.
    await act(async () => setValue(byPlaceholder("Material, pose"), "brand new"));
    expect(snap.caption.compositional_deconstruction.elements[1].desc).toBe("brand new");
  });

  it("keeps the right row expanded by STABLE key when a middle row is removed (sc-8115)", async () => {
    await mount();
    // Add three objects; each add expands the new one, so after this the third
    // is expanded. Tag each so we can identify them by content, not index.
    await act(async () => click(buttonByText("+ Object")));
    await act(async () => setValue(byPlaceholder("Material, pose"), "one"));
    await act(async () => click(buttonByText("+ Object")));
    await act(async () => setValue(byPlaceholder("Material, pose"), "two"));
    await act(async () => click(buttonByText("+ Object")));
    await act(async () => setValue(byPlaceholder("Material, pose"), "three"));

    expect(snap.caption.compositional_deconstruction.elements.map((e) => e.desc)).toEqual([
      "one",
      "two",
      "three",
    ]);

    // Expand the FIRST row ("one") so the expanded key points at element index 0.
    const oneSummary = summaryButtons().find((b) =>
      b.querySelector(".structured-element-summary-text").textContent.includes("one"),
    );
    await act(async () => click(oneSummary));
    expect(expandedRows()[0].querySelector("textarea").value).toBe("one");

    // Remove the MIDDLE row ("two"). The expanded element was "one" (index 0),
    // which is NOT the removed row — if expansion were tracked by index, the
    // remaining "one" (still index 0) would stay expanded only by luck; the real
    // proof is that after removing the middle, "one" is still the expanded one.
    const twoRow = elementRows().find(
      (r) => r.querySelector(".structured-element-summary-text")?.textContent.includes("two"),
    );
    const removeTwo = [...twoRow.querySelectorAll("button")].find((b) => b.textContent.trim() === "Remove");
    await act(async () => click(removeTwo));

    expect(snap.caption.compositional_deconstruction.elements.map((e) => e.desc)).toEqual(["one", "three"]);
    expect(expandedRows()).toHaveLength(1);
    expect(expandedRows()[0].querySelector("textarea").value).toBe("one");
  });

  it("keeps a LATER expanded row open by STABLE key when an EARLIER middle row is removed (sc-8115)", async () => {
    // This is the distinguishing case for stable-key vs index tracking. The
    // earlier "...middle row is removed" test expands index 0 and removes index
    // 1, so an index impl would keep index 0 expanded too — it can't tell the
    // implementations apart. Here we expand a LATER row and remove an EARLIER
    // one, so the surviving expanded element SHIFTS index: an index impl would
    // leave the wrong row (or nothing) expanded; only stable-key keeps "three".
    await mount();
    await act(async () => click(buttonByText("+ Object")));
    await act(async () => setValue(byPlaceholder("Material, pose"), "one"));
    await act(async () => click(buttonByText("+ Object")));
    await act(async () => setValue(byPlaceholder("Material, pose"), "two"));
    await act(async () => click(buttonByText("+ Object")));
    await act(async () => setValue(byPlaceholder("Material, pose"), "three"));

    // The third row ("three", index 2) is expanded — each add auto-expands the
    // newest row, so we don't need to click anything.
    expect(snap.caption.compositional_deconstruction.elements.map((e) => e.desc)).toEqual([
      "one",
      "two",
      "three",
    ]);
    expect(expandedRows()).toHaveLength(1);
    expect(expandedRows()[0].querySelector("textarea").value).toBe("three");

    // Remove the EARLIER middle row ("two", index 1). This shifts "three" from
    // index 2 down to index 1. The removed row is NOT the expanded one, so the
    // expanded element must remain "three".
    const twoRow = elementRows().find(
      (r) => r.querySelector(".structured-element-summary-text")?.textContent.includes("two"),
    );
    const removeTwo = [...twoRow.querySelectorAll("button")].find((b) => b.textContent.trim() === "Remove");
    await act(async () => click(removeTwo));

    expect(snap.caption.compositional_deconstruction.elements.map((e) => e.desc)).toEqual(["one", "three"]);
    expect(expandedRows()).toHaveLength(1);
    // With an index impl, expandedIndex 2 would now be out of range (only
    // indices 0 and 1 remain) or point at the wrong row; only the stable-key
    // impl keeps the SHIFTED "three" row expanded.
    expect(expandedRows()[0].querySelector("textarea").value).toBe("three");
  });

  it("falls back to a neighbour when the EXPANDED row is removed (sc-8115)", async () => {
    await mount({
      initialCaption: captionWithElements([
        { type: "obj", desc: "alpha" },
        { type: "obj", desc: "beta" },
        { type: "obj", desc: "gamma" },
      ]),
    });
    // Default expanded is the last ("gamma"). Remove it.
    const gammaRow = elementRows().find((r) => r.querySelector("textarea")?.value === "gamma");
    const removeGamma = [...gammaRow.querySelectorAll("button")].find((b) => b.textContent.trim() === "Remove");
    await act(async () => click(removeGamma));

    // A neighbour stays expanded (the previous row, "beta"), never zero/orphaned.
    expect(snap.caption.compositional_deconstruction.elements.map((e) => e.desc)).toEqual(["alpha", "beta"]);
    expect(expandedRows()).toHaveLength(1);
    expect(expandedRows()[0].querySelector("textarea").value).toBe("beta");
  });

  it("expands the last freshly-minted element after an external (magic-prompt) expand (sc-8115)", async () => {
    const expanded = {
      high_level_description: "a red fox",
      compositional_deconstruction: {
        background: "snow",
        elements: [
          { type: "obj", desc: "a red fox" },
          { type: "text", text: "WINTER", desc: "title" },
        ],
      },
    };
    const onMagicExpand = vi.fn(async () => expanded);
    await mount({ initialMode: "plain", initialPlain: "a red fox in the snow", onMagicExpand });
    await clickAndSettle(buttonByText("✨ Expand to caption"));

    // Dropped into the builder with both elements present, exactly one expanded
    // (the last freshly-minted row), the rest collapsed to summaries.
    expect(snap.mode).toBe("form");
    expect(elementRows()).toHaveLength(2);
    expect(expandedRows()).toHaveLength(1);
    expect(expandedRows()[0].querySelector("textarea").value).toBe("title");
  });

  it("collapses everything when the open row is collapsed via its toggle (sc-8115)", async () => {
    await mount({
      initialCaption: captionWithElements([{ type: "obj", desc: "solo" }]),
    });
    expect(expandedRows()).toHaveLength(1);
    // Click the expanded row's toggle to collapse it -> zero expanded.
    const toggle = container.querySelector(".structured-element-toggle");
    await act(async () => click(toggle));
    expect(expandedRows()).toHaveLength(0);
    expect(summaryButtons()).toHaveLength(1);
  });
});
