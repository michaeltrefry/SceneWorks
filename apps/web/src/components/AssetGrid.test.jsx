import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AssetCard, AssetGrid } from "./assetPanels.jsx";

const assets = [
  { id: "a", type: "image", displayName: "one.png", file: { path: "assets/one.png", mimeType: "image/png" }, projectId: "p1" },
  { id: "b", type: "image", displayName: "two.png", file: { path: "assets/two.png", mimeType: "image/png" }, projectId: "p1" },
];

describe("AssetGrid multi-select (sc-6112)", () => {
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

  const tiles = () => [...container.querySelectorAll(".asset-tile")];
  const checks = () => [...container.querySelectorAll(".asset-tile-check input")];

  it("single-select mode (no onToggleSelect) renders no checkboxes and selects on tile click", async () => {
    const setSelectedAssetId = vi.fn();
    await act(() => {
      root.render(<AssetGrid assets={assets} onPreview={vi.fn()} selectedAsset={null} setSelectedAssetId={setSelectedAssetId} />);
    });
    expect(checks()).toHaveLength(0);
    await act(async () => tiles()[0].click());
    expect(setSelectedAssetId).toHaveBeenCalledWith("a");
  });

  it("multi-select mode renders a checkbox per tile, reflects selectedIds, and toggles without single-selecting", async () => {
    const onToggleSelect = vi.fn();
    const setSelectedAssetId = vi.fn();
    await act(() => {
      root.render(
        <AssetGrid
          assets={assets}
          onPreview={vi.fn()}
          selectedAsset={null}
          setSelectedAssetId={setSelectedAssetId}
          selectedIds={new Set(["b"])}
          onToggleSelect={onToggleSelect}
        />,
      );
    });
    const boxes = checks();
    expect(boxes).toHaveLength(2);
    expect(boxes[0].checked).toBe(false);
    expect(boxes[1].checked).toBe(true); // "b" is in selectedIds
    expect(container.querySelectorAll(".asset-tile-wrap.selected")).toHaveLength(1);

    // Toggling the checkbox calls onToggleSelect, not the single-select handler.
    await act(async () => boxes[0].click());
    expect(onToggleSelect).toHaveBeenCalledWith("a");
    expect(setSelectedAssetId).not.toHaveBeenCalled();

    // The tile body still drives single-select (the detail flow is unchanged).
    await act(async () => tiles()[0].click());
    expect(setSelectedAssetId).toHaveBeenCalledWith("a");
  });

  it("suppresses the native context menu on a Library grid thumbnail cell (sc-8731) without breaking selection", async () => {
    const setSelectedAssetId = vi.fn();
    await act(() => {
      root.render(<AssetGrid assets={assets} onPreview={vi.fn()} selectedAsset={null} setSelectedAssetId={setSelectedAssetId} />);
    });
    const tile = tiles()[0];
    const event = new MouseEvent("contextmenu", { bubbles: true, cancelable: true });
    tile.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);

    // Suppressing the contextmenu must not disturb the left-click select flow.
    await act(async () => tile.click());
    expect(setSelectedAssetId).toHaveBeenCalledWith("a");
  });
});

describe("AssetCard native context-menu suppression (sc-8731)", () => {
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

  it("suppresses the native menu on the studio AssetCard thumbnail without breaking open-preview", async () => {
    const onPreview = vi.fn();
    await act(() => {
      root.render(
        <AssetCard
          asset={assets[0]}
          deleteAsset={vi.fn()}
          purgeAsset={vi.fn()}
          onPreview={onPreview}
          updateAssetStatus={vi.fn()}
        />,
      );
    });
    const preview = container.querySelector(".preview-button");
    expect(preview).not.toBeNull();
    const event = new MouseEvent("contextmenu", { bubbles: true, cancelable: true });
    preview.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);

    // Left-click still opens the preview.
    await act(async () => preview.click());
    expect(onPreview).toHaveBeenCalledWith(assets[0]);
  });
});
