import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Characters } from "./Characters.jsx";
import { AppContext } from "../../context/AppContext.js";

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "project-a", name: "Project A" },
    characters: [{ id: "char_1", name: "Mara", references: [] }],
    createCharacter: vi.fn(),
    addCharacterReference: vi.fn(),
    updateCharacter: vi.fn(),
    importAsset: vi.fn(),
    imageModels: [],
    createImageJob: vi.fn(),
    setUiMode: vi.fn(),
    setActiveView: vi.fn(),
    ...overrides,
  };
}

async function render(value) {
  await act(async () => {
    root.render(
      <AppContext.Provider value={value}>
        <Characters />
      </AppContext.Provider>,
    );
  });
}

describe("Characters", () => {
  it("routes the Advanced affordance to the advanced Character Studio", async () => {
    const ctx = baseContext();
    await render(ctx);

    const link = [...container.querySelectorAll(".sw-advlink")].find((node) => node.textContent.includes("Use Advanced"));
    await act(async () => {
      link.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });

    expect(ctx.setUiMode).toHaveBeenCalledWith("advanced");
    expect(ctx.setActiveView).toHaveBeenCalledWith("Characters");
  });
});
