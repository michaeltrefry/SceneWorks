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
  function Harness({ initialMode = "form", initialCaption, initialPlain = "", onMagicExpand, magicModelMissing = false, onDownloadMagicModel }) {
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
});
