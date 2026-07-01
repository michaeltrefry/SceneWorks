// sc-8810: media URLs must authenticate in remote-auth mode. Element-driven
// requests (<img src>, <video src>, <a download>) cannot send the token header,
// so every media-URL producer appends the short-lived media ticket minted from
// POST /api/v1/files/ticket. These tests pin the central helper and the
// producers that ride on it (assetUrl/posterUrl, DocumentReader's fetch).
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { API_BASE_URL, getMediaTicket, setMediaTicket, withMediaTicket } from "./api.js";
import { assetUrl, posterUrl } from "./components/assetMedia.jsx";
import { AssetDetail } from "./components/assetPanels.jsx";
import { keypointSourceImageUrl } from "./keypointLibrary.js";

const imageAsset = {
  id: "img",
  type: "image",
  displayName: "one.png",
  file: { path: "assets/one.png", mimeType: "image/png" },
  projectId: "p1",
};

const videoAsset = {
  id: "vid",
  type: "video",
  displayName: "clip.mp4",
  file: { path: "assets/clip.mp4", mimeType: "video/mp4" },
  projectId: "p1",
};

afterEach(() => {
  setMediaTicket("");
});

describe("withMediaTicket", () => {
  it("is a no-op while no ticket is set (desktop/loopback and auth-off modes)", () => {
    setMediaTicket("");
    expect(withMediaTicket("http://x/api/v1/projects/p1/files/a.png")).toBe(
      "http://x/api/v1/projects/p1/files/a.png",
    );
    expect(withMediaTicket("")).toBe("");
  });

  it("appends the ticket as a query param, joining on existing query strings", () => {
    setMediaTicket("abc123");
    expect(getMediaTicket()).toBe("abc123");
    expect(withMediaTicket("http://x/files/a.png")).toBe("http://x/files/a.png?ticket=abc123");
    expect(withMediaTicket("http://x/files/a.png?v=2")).toBe("http://x/files/a.png?v=2&ticket=abc123");
  });

  it("leaves empty URLs alone even with a ticket set", () => {
    setMediaTicket("abc123");
    expect(withMediaTicket("")).toBe("");
    expect(withMediaTicket(null)).toBe(null);
  });
});

describe("asset URL producers carry the media ticket", () => {
  it("assetUrl appends the ticket to file-path URLs", () => {
    setMediaTicket("t0ken");
    expect(assetUrl(imageAsset)).toBe(
      `${API_BASE_URL}/api/v1/projects/p1/files/assets/one.png?ticket=t0ken`,
    );
  });

  it("assetUrl appends the ticket to normalized `url` assets", () => {
    setMediaTicket("t0ken");
    expect(assetUrl({ ...imageAsset, url: "/api/v1/projects/p1/files/assets/one.png" })).toBe(
      `${API_BASE_URL}/api/v1/projects/p1/files/assets/one.png?ticket=t0ken`,
    );
  });

  it("assetUrl stays bare without a ticket", () => {
    expect(assetUrl(imageAsset)).toBe(`${API_BASE_URL}/api/v1/projects/p1/files/assets/one.png`);
  });

  it("posterUrl swaps the extension BEFORE the ticket query string", () => {
    setMediaTicket("t0ken");
    expect(posterUrl(videoAsset)).toBe(
      `${API_BASE_URL}/api/v1/projects/p1/files/assets/clip.poster.jpg?ticket=t0ken`,
    );
  });

  it("keypointSourceImageUrl appends the ticket", () => {
    setMediaTicket("t0ken");
    const url = keypointSourceImageUrl("assets/keypoints/asset_x.png");
    expect(url).toContain("/files/assets/keypoints/asset_x.png?ticket=t0ken");
  });
});

describe("DocumentReader authenticates its document fetch (sc-8810)", () => {
  let container;
  let root;
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    global.fetch = originalFetch;
  });

  it("fetches the document through a ticketed URL", async () => {
    setMediaTicket("d0cticket");
    global.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ segments: [{ type: "text", text: "hello" }] }),
    }));
    const documentAsset = {
      id: "doc",
      type: "document",
      displayName: "story.json",
      file: { path: "assets/documents/story.json", mimeType: "application/json" },
      projectId: "p1",
    };
    await act(async () =>
      root.render(
        <AssetDetail
          asset={documentAsset}
          deleteAsset={() => {}}
          purgeAsset={() => {}}
          onPreview={() => {}}
          onSendImage={() => {}}
          onSendVideo={() => {}}
          onSendEditor={() => {}}
          updateAssetStatus={() => {}}
          updateAssetTags={() => {}}
          availableTags={[]}
        />,
      ),
    );
    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url] = global.fetch.mock.calls[0];
    expect(url).toBe(
      `${API_BASE_URL}/api/v1/projects/p1/files/assets/documents/story.json?ticket=d0cticket`,
    );
    expect(container.textContent).toContain("hello");
  });
});
