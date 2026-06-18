import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useLookExemplars } from "./useLookExemplars.js";
import { LOOK_EXEMPLARS_STORAGE_KEY } from "./lookExemplars.js";
import { AppContext } from "../../context/AppContext.js";

function Harness({ preferredModelId = null }) {
  const { assetForLook, refresh, refreshing, canRender, hasAny, pending } = useLookExemplars(preferredModelId);
  return (
    <div>
      <span data-testid="can">{String(canRender)}</span>
      <span data-testid="any">{String(hasAny)}</span>
      <span data-testid="refreshing">{String(refreshing)}</span>
      <span data-testid="photo-url">{assetForLook("photo")?.url ?? ""}</span>
      <span data-testid="photo-pending">{String(Boolean(pending.photo))}</span>
      <button onClick={() => refresh()}>refresh-all</button>
      <button onClick={() => refresh(["photo"])}>refresh-photo</button>
    </div>
  );
}

let container;
let root;

beforeEach(() => {
  localStorage.clear();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  localStorage.clear();
});

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function render(value, props = {}) {
  return act(() => {
    root.render(
      <AppContext.Provider value={value}>
        <Harness {...props} />
      </AppContext.Provider>,
    );
  });
}

function text(testid) {
  return container.querySelector(`[data-testid="${testid}"]`).textContent;
}

function findButton(label) {
  return [...container.querySelectorAll("button")].find((button) => button.textContent === label);
}

function baseContext(overrides = {}) {
  return {
    activeProject: { id: "proj-1", name: "Proj" },
    imageModels: [{ id: "z_image_turbo", capabilities: ["text_to_image"] }],
    createImageJob: vi.fn(async () => ({ id: "job_photo" })),
    jobs: [],
    recentImageAssets: [],
    mediaAssets: [],
    ...overrides,
  };
}

describe("useLookExemplars", () => {
  it("can render only with a model and an active project", async () => {
    await render(baseContext({ imageModels: [] }));
    expect(text("can")).toBe("false");
    await render(baseContext());
    expect(text("can")).toBe("true");
    expect(text("any")).toBe("false");
  });

  it("submits one render per look and resolves the asset back to the look", async () => {
    const ctx = baseContext();
    await render(ctx);

    await act(() => findButton("refresh-photo").dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    await settle();

    expect(ctx.createImageJob).toHaveBeenCalledTimes(1);
    const payload = ctx.createImageJob.mock.calls[0][0];
    expect(payload).toMatchObject({ mode: "text_to_image", count: 1, model: "z_image_turbo" });
    expect(typeof payload.seed).toBe("number");
    expect(payload.prompt).toContain("professional photograph");
    expect(text("photo-pending")).toBe("true");

    // The render job completes and its asset lands in the recent list.
    const asset = { id: "asset-1", type: "image", url: "/api/v1/projects/proj-1/files/photo.png", lineage: { jobId: "job_photo" } };
    await render({
      ...ctx,
      jobs: [{ id: "job_photo", status: "completed", result: { assetIds: ["asset-1"], seed: 7 } }],
      recentImageAssets: [asset],
    });
    await settle();

    expect(text("photo-url")).toBe("/api/v1/projects/proj-1/files/photo.png");
    expect(text("photo-pending")).toBe("false");
    const stored = JSON.parse(localStorage.getItem(LOOK_EXEMPLARS_STORAGE_KEY));
    expect(stored["proj-1"].z_image_turbo.photo).toMatchObject({ assetId: "asset-1" });
  });

  it("rehydrates cached exemplars from storage on mount", async () => {
    localStorage.setItem(
      LOOK_EXEMPLARS_STORAGE_KEY,
      JSON.stringify({
        "proj-1": {
          z_image_turbo: {
            photo: { assetId: "asset-9", url: "/api/v1/projects/proj-1/files/cached.png", seed: 1 },
          },
        },
      }),
    );
    await render(baseContext());
    expect(text("any")).toBe("true");
    expect(text("photo-url")).toBe("/api/v1/projects/proj-1/files/cached.png");
  });

  it("keeps cached exemplars separate per model", async () => {
    localStorage.setItem(
      LOOK_EXEMPLARS_STORAGE_KEY,
      JSON.stringify({
        "proj-1": {
          z_image_turbo: {
            photo: { assetId: "asset-z", url: "/api/v1/projects/proj-1/files/z.png", seed: 1 },
          },
          realvisxl: {
            photo: { assetId: "asset-r", url: "/api/v1/projects/proj-1/files/r.png", seed: 2 },
          },
        },
      }),
    );
    const ctx = baseContext({
      imageModels: [
        { id: "z_image_turbo", capabilities: ["text_to_image"] },
        { id: "realvisxl", capabilities: ["text_to_image"] },
      ],
    });

    await render(ctx, { preferredModelId: "realvisxl" });
    expect(text("photo-url")).toBe("/api/v1/projects/proj-1/files/r.png");

    await render(ctx, { preferredModelId: "z_image_turbo" });
    expect(text("photo-url")).toBe("/api/v1/projects/proj-1/files/z.png");
  });

  it("renders every look when refreshed without an explicit list", async () => {
    let counter = 0;
    const ctx = baseContext({ createImageJob: vi.fn(async () => ({ id: `job_${counter++}` })) });
    await render(ctx);
    await act(() => findButton("refresh-all").dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
    await settle();
    expect(ctx.createImageJob).toHaveBeenCalledTimes(6);
  });
});
