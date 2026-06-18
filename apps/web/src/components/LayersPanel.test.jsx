import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LayersPanel } from "./LayersPanel.jsx";

// Two-layer stack: bottom "Background", top "Sky" (active). Rows render top-first.
const stack = () => [
  { id: "a", name: "Background", objectUrl: "blob:a", visible: true, opacity: 1 },
  { id: "b", name: "Sky", objectUrl: "blob:b", visible: true, opacity: 0.5 },
];

const noopHandlers = () => ({
  onSelect: vi.fn(),
  onToggleVisible: vi.fn(),
  onSetOpacity: vi.fn(),
  onRename: vi.fn(),
  onReorder: vi.fn(),
  onAdd: vi.fn(),
  onDelete: vi.fn(),
  onDuplicate: vi.fn(),
});

describe("LayersPanel (sc-6118)", () => {
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
    vi.clearAllMocks();
  });

  const mount = (props) =>
    act(() => {
      root.render(<LayersPanel layers={stack()} activeLayerId="b" {...noopHandlers()} {...props} />);
    });

  const rows = () => [...container.querySelectorAll('[role="listitem"]')];
  const rowByName = (name) =>
    rows().find((r) => r.querySelector(".image-editor-layer-name")?.textContent === name);

  it("renders the stack top-first, marking the active layer", async () => {
    await mount();
    const names = [...container.querySelectorAll(".image-editor-layer-name")].map((n) => n.textContent);
    expect(names).toEqual(["Sky", "Background"]); // top of stack first
    expect(rowByName("Sky").getAttribute("aria-current")).toBe("true");
    expect(rowByName("Background").getAttribute("aria-current")).toBeNull();
  });

  it("thumbnails use each layer's live object URL", async () => {
    await mount();
    const srcs = [...container.querySelectorAll("img.image-editor-layer-thumb")].map((i) => i.getAttribute("src"));
    expect(srcs).toEqual(["blob:b", "blob:a"]);
  });

  it("clicking a row selects that layer", async () => {
    const h = noopHandlers();
    await act(() => root.render(<LayersPanel layers={stack()} activeLayerId="b" {...h} />));
    await act(() => rowByName("Background").click());
    expect(h.onSelect).toHaveBeenCalledWith("a");
  });

  it("the visibility button toggles + reflects state, without selecting the row", async () => {
    const h = noopHandlers();
    const layers = stack();
    layers[0].visible = false; // Background hidden
    await act(() => root.render(<LayersPanel layers={layers} activeLayerId="b" {...h} />));
    const bg = rowByName("Background");
    const vis = bg.querySelector(".image-editor-layer-vis");
    expect(vis.getAttribute("aria-pressed")).toBe("false");
    expect(vis.textContent).toBe("○");
    await act(() => vis.click());
    expect(h.onToggleVisible).toHaveBeenCalledWith("a");
    expect(h.onSelect).not.toHaveBeenCalled(); // stopPropagation
  });

  it("opacity slider flags the first change of a gesture as a checkpoint, the rest not", async () => {
    const h = noopHandlers();
    await act(() => root.render(<LayersPanel layers={stack()} activeLayerId="b" {...h} />));
    const slider = rowByName("Sky").querySelector('input[type="range"]');
    const nativeSet = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    const setVal = (v) =>
      act(() => {
        nativeSet.call(slider, String(v)); // React tracks the controlled value
        slider.dispatchEvent(new Event("input", { bubbles: true }));
      });
    await setVal(40);
    await setVal(30);
    expect(h.onSetOpacity).toHaveBeenNthCalledWith(1, "b", 0.4, true); // gesture start
    expect(h.onSetOpacity).toHaveBeenNthCalledWith(2, "b", 0.3, false);
    // Releasing ends the gesture → the next change starts a fresh checkpoint.
    await act(() => slider.dispatchEvent(new Event("pointerup", { bubbles: true })));
    await setVal(20);
    expect(h.onSetOpacity).toHaveBeenNthCalledWith(3, "b", 0.2, true);
  });

  it("double-click → rename input → Enter commits; Escape cancels", async () => {
    const h = noopHandlers();
    await act(() => root.render(<LayersPanel layers={stack()} activeLayerId="b" {...h} />));
    await act(() => rowByName("Sky").querySelector(".image-editor-layer-name").dispatchEvent(new MouseEvent("dblclick", { bubbles: true })));
    let input = container.querySelector(".image-editor-layer-rename");
    expect(input).toBeTruthy();
    // React tracks the controlled value, so set it via the native setter (a direct
    // `input.value =` is ignored by React's change tracker).
    const setInputValue = (el, value) => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(el, value);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    };
    await act(() => setInputValue(input, "Clouds"));
    await act(() => input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })));
    expect(h.onRename).toHaveBeenCalledWith("b", "Clouds");

    // Escape discards.
    await act(() => rowByName("Background").querySelector(".image-editor-layer-name").dispatchEvent(new MouseEvent("dblclick", { bubbles: true })));
    input = container.querySelector(".image-editor-layer-rename");
    await act(() => input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
    expect(h.onRename).toHaveBeenCalledTimes(1); // not called again
  });

  it("add / duplicate / delete wire to their handlers", async () => {
    const h = noopHandlers();
    await act(() => root.render(<LayersPanel layers={stack()} activeLayerId="b" {...h} />));
    await act(() => container.querySelector(".image-editor-layers-add").click());
    expect(h.onAdd).toHaveBeenCalled();
    await act(() => rowByName("Sky").querySelector('[aria-label="Duplicate layer"]').click());
    expect(h.onDuplicate).toHaveBeenCalledWith("b");
    await act(() => rowByName("Background").querySelector('[aria-label="Delete layer"]').click());
    expect(h.onDelete).toHaveBeenCalledWith("a");
  });

  it("delete is disabled when only one layer remains", async () => {
    await act(() =>
      root.render(
        <LayersPanel
          layers={[{ id: "a", name: "Background", objectUrl: "blob:a", visible: true, opacity: 1 }]}
          activeLayerId="a"
          {...noopHandlers()}
        />,
      ),
    );
    expect(container.querySelector('[aria-label="Delete layer"]').disabled).toBe(true);
  });

  it("reorder buttons move within the stack and disable at the ends", async () => {
    const h = noopHandlers();
    await act(() => root.render(<LayersPanel layers={stack()} activeLayerId="b" {...h} />));
    // Top row "Sky" (stack index 1): up disabled, down enabled.
    const sky = rowByName("Sky");
    expect(sky.querySelector('[aria-label="Move layer up"]').disabled).toBe(true);
    await act(() => sky.querySelector('[aria-label="Move layer down"]').click());
    expect(h.onReorder).toHaveBeenCalledWith("b", 0);
    // Bottom row "Background" (index 0): down disabled, up enabled.
    const bg = rowByName("Background");
    expect(bg.querySelector('[aria-label="Move layer down"]').disabled).toBe(true);
    await act(() => bg.querySelector('[aria-label="Move layer up"]').click());
    expect(h.onReorder).toHaveBeenCalledWith("a", 1);
  });

  it("busy disables structural ops (add / reorder / duplicate / delete)", async () => {
    await act(() => root.render(<LayersPanel layers={stack()} activeLayerId="b" busy {...noopHandlers()} />));
    expect(container.querySelector(".image-editor-layers-add").disabled).toBe(true);
    for (const op of container.querySelectorAll(".image-editor-layer-op")) {
      expect(op.disabled).toBe(true);
    }
  });
});
