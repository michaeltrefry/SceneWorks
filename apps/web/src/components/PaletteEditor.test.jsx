import React, { act, useState } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import PaletteEditor from "./PaletteEditor.jsx";

function setValue(el, value) {
  const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value").set.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
}
function click(el) {
  el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
}

describe("PaletteEditor", () => {
  let container;
  let root;
  let last;

  function Harness({ initial = null, max = 16 }) {
    const [value, setVal] = useState(initial);
    last = value;
    return (
      <PaletteEditor
        value={value}
        max={max}
        label="Color palette"
        onChange={(next) => {
          last = next;
          setVal(next);
        }}
      />
    );
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    last = undefined;
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.clearAllMocks();
  });

  const hexInput = () => container.querySelector('input[aria-label="Hex color"]');
  const addBtn = () => [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === "Add");

  async function mount(props = {}) {
    await act(async () => root.render(<Harness {...props} />));
  }

  it("adds a lowercase hex as a normalized uppercase swatch", async () => {
    await mount();
    await act(async () => setValue(hexInput(), "#abcdef"));
    await act(async () => click(addBtn()));
    expect(last).toEqual(["#ABCDEF"]);
    expect([...container.querySelectorAll(".palette-swatch-hex")].map((s) => s.textContent)).toEqual(["#ABCDEF"]);
  });

  it("preserves insertion order across adds", async () => {
    await mount();
    for (const c of ["#111111", "#222222", "#333333"]) {
      await act(async () => setValue(hexInput(), c));
      await act(async () => click(addBtn()));
    }
    expect(last).toEqual(["#111111", "#222222", "#333333"]);
  });

  it("disables Add for a duplicate color", async () => {
    await mount({ initial: ["#FF0000"] });
    await act(async () => setValue(hexInput(), "#ff0000"));
    expect(addBtn().disabled).toBe(true);
  });

  it("disables Add for an invalid hex", async () => {
    await mount();
    await act(async () => setValue(hexInput(), "not-a-color"));
    expect(addBtn().disabled).toBe(true);
  });

  it("caps at max and disables further input", async () => {
    await mount({ initial: ["#000000", "#111111"], max: 2 });
    expect(addBtn().disabled).toBe(true);
    expect(hexInput().disabled).toBe(true);
    expect(container.textContent).toContain("Maximum 2 colors");
  });

  it("removes a swatch, reporting null when the last one is removed", async () => {
    await mount({ initial: ["#FF0000"] });
    const remove = container.querySelector('button[aria-label="Remove #FF0000"]');
    await act(async () => click(remove));
    expect(last).toBe(null);
  });
});
