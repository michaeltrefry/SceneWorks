import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BatchOperationsPanel } from "./BatchOperationsPanel.jsx";

const assets = [
  { id: "a", displayName: "one.png" },
  { id: "b", displayName: "two.png" },
];
const upscaleEngines = [
  { key: "real-esrgan", label: "Real-ESRGAN", factors: [2, 4] },
  { key: "seedvr2", label: "SeedVR2", factors: [2, 4], softness: true },
];

// React controlled inputs ignore a direct `.value =`; drive them via the native setter.
function setValue(el, value) {
  const proto = el instanceof window.HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value").set.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("BatchOperationsPanel (sc-6112)", () => {
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
      root.render(
        <BatchOperationsPanel
          assets={assets}
          upscaleEngines={upscaleEngines}
          editModels={[{ id: "m1", label: "Edit Model" }]}
          detailModels={[{ id: "d1", label: "Detail Model" }]}
          onRun={vi.fn()}
          onClose={vi.fn()}
          {...props}
        />,
      );
    });

  const tab = (label) => [...document.body.querySelectorAll(".batch-ops-tabs button")].find((b) => b.textContent.trim() === label);
  const runButton = () => [...document.body.querySelectorAll(".batch-ops-actions button")].find((b) => /^Run on/.test(b.textContent));

  it("shows the op tabs + upscale params, and the asset count in the head", async () => {
    await mount();
    expect(document.body.querySelector(".batch-ops-head h3").textContent).toContain("2 images");
    expect([...document.body.querySelectorAll(".batch-ops-tabs button")].map((b) => b.textContent.trim())).toEqual([
      "Upscale",
      "Detail enhance",
      "AI edit",
    ]);
    // Default op is upscale → an Engine + Factor select.
    expect(document.body.querySelector(".batch-ops-params").textContent).toContain("Engine");
    expect(document.body.querySelector(".batch-ops-params").textContent).toContain("Factor");
  });

  it("fans the upscale op + params back to onRun", async () => {
    const onRun = vi.fn();
    await mount({ onRun });
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenCalledTimes(1);
    const [op, params] = onRun.mock.calls[0];
    expect(op).toBe("upscale");
    expect(params).toMatchObject({ engine: "real-esrgan", factor: 4 });
  });

  it("requires a prompt for the edit op, then passes the prompt + model", async () => {
    const onRun = vi.fn();
    await mount({ onRun });
    await act(async () => tab("AI edit").click());
    // Empty prompt → Run disabled.
    expect(runButton().disabled).toBe(true);
    const textarea = document.body.querySelector(".batch-ops-params textarea");
    await act(async () => setValue(textarea, "make it snow"));
    expect(runButton().disabled).toBe(false);
    await act(async () => runButton().click());
    expect(onRun).toHaveBeenCalledWith("edit", expect.objectContaining({ model: "m1", prompt: "make it snow" }));
  });

  it("renders the per-item progress view (with statuses) when items are supplied", async () => {
    await mount({
      items: [
        { asset: { id: "a", displayName: "one.png" }, status: "completed" },
        { asset: { id: "b", displayName: "two.png" }, status: "running" },
      ],
      progress: { total: 2, done: 1, failed: 0, completed: 1, running: 1, queued: 0, allDone: false },
    });
    // Progress view replaces the form (no Run button) and lists per-item status.
    expect(runButton()).toBeUndefined();
    expect(document.body.querySelector(".batch-ops-summary").textContent).toContain("1 / 2 done");
    const items = [...document.body.querySelectorAll(".batch-ops-item")];
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toContain("Done");
    expect(items[1].textContent).toContain("Running");
  });
});
