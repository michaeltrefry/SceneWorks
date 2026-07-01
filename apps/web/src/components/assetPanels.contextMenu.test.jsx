import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// sc-8729: FullscreenPreview's custom right-click context menu + Save As button.
//
// The menu's visibility of "Reveal" depends on `isDesktop`, which assetPanels.jsx
// imports from runtime.js at module load. To exercise both desktop and browser
// variants we mock runtime.js with a mutable flag and re-import the component fresh
// per suite (vi.resetModules). The Save As / Reveal actions are mocked from
// assetActions.js so we assert the menu wires to them without touching Tauri.

const actionMocks = vi.hoisted(() => ({
  saveAssetAs: vi.fn(),
  revealAsset: vi.fn(),
}));

vi.mock("../assetActions.js", () => ({
  saveAssetAs: actionMocks.saveAssetAs,
  revealAsset: actionMocks.revealAsset,
}));

// Mutable desktop flag driven per-suite. The component reads `isDesktop` at module
// load, so tests set this BEFORE importing the component (below).
const runtimeState = vi.hoisted(() => ({ isDesktop: true }));

vi.mock("../runtime.js", () => ({
  get isDesktop() {
    return runtimeState.isDesktop;
  },
  tauriInvoke: vi.fn(),
}));

const imageAsset = {
  id: "asset-img",
  projectId: "project-1",
  displayName: "Plate",
  type: "image",
  status: {},
  file: { path: "assets/images/plate.png", mimeType: "image/png" },
};

const videoAsset = {
  id: "asset-vid",
  projectId: "project-1",
  displayName: "Clip",
  type: "video",
  status: {},
  file: { path: "assets/videos/clip.mp4", mimeType: "video/mp4" },
};

let container;
let root;
let FullscreenPreview;

const noop = () => {};

function baseProps(overrides = {}) {
  return {
    deleteAsset: noop,
    nextAsset: null,
    onClose: noop,
    onPreviewAsset: noop,
    previousAsset: null,
    purgeAsset: noop,
    updateAssetStatus: noop,
    ...overrides,
  };
}

async function renderPreview(props) {
  root = createRoot(container);
  await act(async () => {
    root.render(<FullscreenPreview {...props} />);
  });
}

// Right-click the preview stage and return the resulting contextmenu event so the
// caller can assert defaultPrevented.
async function rightClickStage() {
  const stage = document.body.querySelector(".preview-modal-stage");
  const event = new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: 120, clientY: 90 });
  await act(async () => {
    stage.dispatchEvent(event);
  });
  return event;
}

function menu() {
  return document.body.querySelector(".preview-context-menu");
}

function menuItemLabels() {
  return [...document.body.querySelectorAll(".preview-context-menu > .preview-context-menu-item")].map((b) =>
    b.textContent.trim(),
  );
}

async function loadComponent(isDesktop) {
  runtimeState.isDesktop = isDesktop;
  vi.resetModules();
  ({ FullscreenPreview } = await import("./assetPanels.jsx"));
}

beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement("div");
  document.body.appendChild(container);
  actionMocks.saveAssetAs.mockReset();
  actionMocks.revealAsset.mockReset();
});

afterEach(() => {
  if (root) {
    act(() => root.unmount());
    root = null;
  }
  container.remove();
});

describe("FullscreenPreview context menu (desktop)", () => {
  beforeEach(async () => {
    await loadComponent(true);
  });

  it("right-clicking an image shows the full menu and suppresses the native menu", async () => {
    const onEditImage = vi.fn();
    const onEditInStudio = vi.fn();
    await renderPreview(baseProps({ asset: imageAsset, onEditImage, onEditInStudio }));

    const event = await rightClickStage();
    expect(event.defaultPrevented).toBe(true);

    expect(menu()).not.toBeNull();
    const labels = menuItemLabels();
    expect(labels).toContain("Save As…");
    expect(labels).toContain("Reveal in Finder/Explorer");
    expect(labels).toContain("Zoom In");
    expect(labels).toContain("Zoom Out");
    expect(labels).toContain("Fit to View");
    // The submenu trigger is its own row.
    expect(document.body.querySelector(".preview-context-menu-submenu-trigger").textContent).toContain("Edit in");
  });

  it("Save As and Reveal menu items call the shared action layer", async () => {
    await renderPreview(baseProps({ asset: imageAsset }));
    await rightClickStage();

    await act(async () => {
      [...document.body.querySelectorAll(".preview-context-menu-item")]
        .find((b) => b.textContent.trim() === "Save As…")
        .click();
    });
    expect(actionMocks.saveAssetAs).toHaveBeenCalledWith(imageAsset);
    // Action closes the menu.
    expect(menu()).toBeNull();

    await rightClickStage();
    await act(async () => {
      [...document.body.querySelectorAll(".preview-context-menu-item")]
        .find((b) => b.textContent.trim() === "Reveal in Finder/Explorer")
        .click();
    });
    expect(actionMocks.revealAsset).toHaveBeenCalledWith(imageAsset);
  });

  it("Zoom In / Zoom Out / Fit menu items drive the view transform", async () => {
    await renderPreview(baseProps({ asset: imageAsset }));
    const transform = () => document.body.querySelector(".preview-zoom-inner").style.transform;
    expect(transform()).toContain("scale(1)");

    await rightClickStage();
    await act(async () => {
      [...document.body.querySelectorAll(".preview-context-menu-item")]
        .find((b) => b.textContent.trim() === "Zoom In")
        .click();
    });
    expect(transform()).not.toContain("scale(1)");

    await rightClickStage();
    await act(async () => {
      [...document.body.querySelectorAll(".preview-context-menu-item")]
        .find((b) => b.textContent.trim() === "Fit to View")
        .click();
    });
    expect(transform()).toContain("scale(1)");
  });

  it("Edit in submenu routes to the Image Editor and Image Studio handlers", async () => {
    const onEditImage = vi.fn();
    const onEditInStudio = vi.fn();
    await renderPreview(baseProps({ asset: imageAsset, onEditImage, onEditInStudio }));

    await rightClickStage();
    await act(async () => {
      document.body.querySelector(".preview-context-menu-submenu-trigger").click();
    });
    const submenuButtons = [...document.body.querySelectorAll(".preview-context-menu-submenu-panel .preview-context-menu-item")];
    expect(submenuButtons.map((b) => b.textContent.trim())).toEqual(["Image Editor", "Image Studio"]);

    await act(async () => {
      submenuButtons.find((b) => b.textContent.trim() === "Image Editor").click();
    });
    expect(onEditImage).toHaveBeenCalledWith(imageAsset);

    await rightClickStage();
    await act(async () => {
      document.body.querySelector(".preview-context-menu-submenu-trigger").click();
    });
    await act(async () => {
      [...document.body.querySelectorAll(".preview-context-menu-submenu-panel .preview-context-menu-item")]
        .find((b) => b.textContent.trim() === "Image Studio")
        .click();
    });
    expect(onEditInStudio).toHaveBeenCalledWith(imageAsset);
  });

  it("right-clicking a video shows the variant with no zoom items", async () => {
    await renderPreview(baseProps({ asset: videoAsset, onEditImage: vi.fn(), onEditInStudio: vi.fn() }));

    const event = await rightClickStage();
    expect(event.defaultPrevented).toBe(true);

    const labels = menuItemLabels();
    expect(labels).toContain("Save As…");
    expect(labels).toContain("Reveal in Finder/Explorer");
    expect(labels).not.toContain("Zoom In");
    expect(labels).not.toContain("Zoom Out");
    expect(labels).not.toContain("Fit to View");
    expect(document.body.querySelector(".preview-context-menu-submenu-trigger")).not.toBeNull();
  });

  it("closes on Escape and on outside pointer-down", async () => {
    await renderPreview(baseProps({ asset: imageAsset }));
    await rightClickStage();
    expect(menu()).not.toBeNull();

    await act(async () => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(menu()).toBeNull();

    await rightClickStage();
    expect(menu()).not.toBeNull();
    await act(async () => {
      document.body.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    });
    expect(menu()).toBeNull();
  });

  it("footer Save As button calls saveAssetAs", async () => {
    await renderPreview(baseProps({ asset: imageAsset }));
    const saveButton = [...document.body.querySelectorAll(".preview-actions button")].find(
      (b) => b.textContent.trim() === "Save As…",
    );
    expect(saveButton).not.toBeUndefined();
    await act(async () => {
      saveButton.click();
    });
    expect(actionMocks.saveAssetAs).toHaveBeenCalledWith(imageAsset);
  });
});

describe("FullscreenPreview context menu (browser / LAN)", () => {
  beforeEach(async () => {
    await loadComponent(false);
  });

  it("omits Reveal in browser mode but keeps Save As", async () => {
    await renderPreview(baseProps({ asset: imageAsset }));
    await rightClickStage();
    const labels = menuItemLabels();
    expect(labels).toContain("Save As…");
    expect(labels).not.toContain("Reveal in Finder/Explorer");
  });

  it("keeps the footer Save As button in browser mode", async () => {
    await renderPreview(baseProps({ asset: videoAsset }));
    const saveButton = [...document.body.querySelectorAll(".preview-actions button")].find(
      (b) => b.textContent.trim() === "Save As…",
    );
    expect(saveButton).not.toBeUndefined();
    await act(async () => {
      saveButton.click();
    });
    expect(actionMocks.saveAssetAs).toHaveBeenCalledWith(videoAsset);
  });
});
